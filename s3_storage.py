"""
S3 Storage Service for AI-CAC outputs.
Handles uploading mask and debug files to AWS S3.
"""

import os
from pathlib import Path as FilePath
from typing import Optional, List
import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig


class S3StorageService:
    """
    Manages S3 uploads and URL generation for AI-CAC outputs.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        prefix: str = "RADFLOW/outputs/ai-cac-outputs",
        presigned_url_expiration: int = 3600,
    ):
        """
        Initialize S3 storage service.

        Args:
            bucket_name: S3 bucket name (from env S3_BUCKET_NAME if not provided)
            region: AWS region (from env AWS_REGION if not provided)
            prefix: S3 key prefix for all uploads (default: "RADFLOW/outputs/ai-cac-outputs")
            presigned_url_expiration: Presigned URL expiration in seconds (default: 3600)
        """
        self.bucket_name = bucket_name or os.environ.get("S3_BUCKET_NAME")
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.prefix = prefix
        self.presigned_url_expiration = presigned_url_expiration

        if not self.bucket_name:
            raise ValueError(
                "S3_BUCKET_NAME must be set in environment or passed to constructor"
            )

        # Initialize S3 client
        # Credentials will be loaded from:
        # - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        # - IAM role (if running on EC2/ECS/Lambda)
        # - ~/.aws/credentials file
        self.s3_client = boto3.client("s3", region_name=self.region)
        
        # Transfer config for optimized uploads
        self.transfer_config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,  # 8 MB
            max_concurrency=10,
            multipart_chunksize=8 * 1024 * 1024,
            use_threads=True
        )

        print(f"S3 Storage initialized: bucket={self.bucket_name}, region={self.region}")

    def upload_file(
        self,
        local_file_path: str,
        s3_key: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a file to S3.

        Args:
            local_file_path: Path to local file to upload
            s3_key: S3 key (path) for the uploaded file
            content_type: MIME type (auto-detected if None)

        Returns:
            S3 key of uploaded file

        Raises:
            FileNotFoundError: If local file doesn't exist
            ClientError: If S3 upload fails
        """
        local_path = FilePath(local_file_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        # Auto-detect content type if not provided
        if content_type is None:
            suffix = local_path.suffix.lower()
            content_type_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".csv": "text/csv",
                ".json": "application/json",
                ".txt": "text/plain",
            }
            content_type = content_type_map.get(suffix, "application/octet-stream")

        # Build full S3 key with prefix
        full_s3_key = f"{self.prefix}/{s3_key}".strip("/")

        # Upload to S3
        extra_args = {"ContentType": content_type}

        try:
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                full_s3_key,
                ExtraArgs=extra_args,
                Config=self.transfer_config,
            )
            print(f"Uploaded to S3: s3://{self.bucket_name}/{full_s3_key}")
            return full_s3_key

        except ClientError as e:
            print(f"S3 upload failed: {e}")
            raise

    def upload_directory(
        self,
        local_dir_path: str,
        s3_prefix: str,
    ) -> List[str]:
        """
        Upload all files in a directory to S3.

        Args:
            local_dir_path: Path to local directory
            s3_prefix: S3 prefix for uploaded files

        Returns:
            List of S3 keys for uploaded files
        """
        local_dir = FilePath(local_dir_path)

        if not local_dir.exists() or not local_dir.is_dir():
            raise ValueError(f"Invalid directory: {local_dir_path}")

        uploaded_keys = []

        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                # Build relative path for S3 key
                relative_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative_path.as_posix()}"

                try:
                    full_key = self.upload_file(str(file_path), s3_key)
                    uploaded_keys.append(full_key)
                except Exception as e:
                    print(f"Failed to upload {file_path}: {e}")

        return uploaded_keys

    def generate_presigned_url(self, s3_key: str) -> str:
        """
        Generate a presigned URL for accessing a file in S3.

        Args:
            s3_key: S3 key of the file

        Returns:
            Presigned URL (valid for presigned_url_expiration seconds)
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=self.presigned_url_expiration,
            )
            return url
        except ClientError as e:
            print(f"Failed to generate presigned URL: {e}")
            raise

    def get_public_url(self, s3_key: str) -> str:
        """
        Generate a public S3 URL (only works if bucket/object is public).

        Args:
            s3_key: S3 key of the file

        Returns:
            Public S3 URL
        """
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

    def delete_file(self, s3_key: str) -> None:
        """
        Delete a file from S3.

        Args:
            s3_key: S3 key of the file to delete
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            print(f"Deleted from S3: s3://{self.bucket_name}/{s3_key}")
        except ClientError as e:
            print(f"S3 delete failed: {e}")
            raise

    def delete_directory(self, s3_prefix: str) -> None:
        """
        Delete all files under an S3 prefix.

        Args:
            s3_prefix: S3 prefix to delete
        """
        try:
            # List all objects with the prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=s3_prefix
            )

            if "Contents" in response:
                # Delete all objects
                objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name, Delete={"Objects": objects}
                )
                print(f"Deleted {len(objects)} objects from S3 prefix: {s3_prefix}")
            else:
                print(f"No objects found with prefix: {s3_prefix}")

        except ClientError as e:
            print(f"S3 delete failed: {e}")
            raise
