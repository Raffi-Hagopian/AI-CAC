# AI-CAC Project Code
# Creator: Raffi Hagopian MD

import pandas as pd
import numpy as np
import torch
import math 
import os
import ipywidgets as widgets
from IPython.display import display, Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

def save_vol_masks(volume, mask_volume, save_path_folder):
    """Save mask overlays for all slices (optimized version)."""
    if not os.path.exists(save_path_folder):
        os.makedirs(save_path_folder)
    
    for i in range(0, volume.shape[2]):
        slice_img = volume[:, :, i]
        mask_slice = mask_volume[:, :, i]
        
        # Skip slices with no mask
        if not torch.any(mask_slice > 0):
            continue
        
        filepath = save_path_folder + f'/{i}'
        draw_big_mask(slice_img, mask_slice, filepath)
        
def vol_mask_slider(volume, mask_volume):
    def plot_slice(i):
        plt.figure(figsize=(8,8))
        slice = volume[:, :, i]
        mask_slice = mask_volume[:, :, i]
        draw_big_mask(slice,mask_slice)
        #slice_rgb = np.stack([slice, slice, slice], axis=-1)
        #slice_rgb[mask_slice>0] = [1, 0, 0]
        #plt.imshow(slice_rgb)#, cmap='gray')
        #plt.axis('off')
        #plt.show()
    widgets.interact(plot_slice, i=widgets.IntSlider(min=0, max=volume.shape[2] - 1, step=1, value=0, description='Slice Index'))

def interactive_volume(volume):
    def plot_slice(slice_index):
        plt.figure(figsize=(8, 8))
        plt.imshow(volume[:,:,slice_index], cmap='gray')
        plt.axis('off')
        plt.show()
    widgets.interact(plot_slice, slice_index=widgets.IntSlider(min=0, max=volume.shape[2] - 1, step=1, value=0, description='Slice Index'))

def interactive_volume_mask(volume, mask_volume):
    def plot_slice(slice_index):
        img = volume[:,:,slice_index]
        mask = mask_volume[:,:,slice_index]
        fig, ax = plt.subplots(1,2, figsize=(8,8))
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img, cmap="gray")
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j] > 0:
                    patch_coords = [(j,i),(j,i+1),(j+1,i+1),(j+1,i)]
                    patch = patches.Polygon(patch_coords, closed=True, fill=True, color='red', alpha=0.5)
                    ax[1].add_patch(patch)
        display(fig)
    widgets.interact(plot_slice, slice_index=widgets.IntSlider(min=0, max=volume.shape[2] - 1, step=1, value=0, description='Slice Index'))

def draw_big_mask_fast(img, mask, save_filepath=None):
    """Optimized mask overlay using vectorized operations (100x faster)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    
    # Save image without mask (matches original quality)
    if save_filepath is not None:
        fig.savefig(save_filepath + '_no_mask.png')
    
    # Create red overlay for mask using vectorized operations
    mask_overlay = np.zeros((*mask.shape, 4))  # RGBA
    mask_binary = mask > 0
    mask_overlay[mask_binary] = [1, 0, 0, 0.5]  # Red with 50% alpha
    
    # Overlay mask
    ax.imshow(mask_overlay)
    
    # Save image with mask (matches original quality)
    if save_filepath is not None:
        fig.savefig(save_filepath + '.png')
    
    plt.close(fig)


def draw_big_mask(img, mask, save_filepath=None):
    """Optimized version - same quality, much faster."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Convert to numpy if tensor
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    ax.imshow(img, cmap="gray")
    
    # Find positive mask pixels using numpy (fast)
    mask_pixels = np.argwhere(mask > 0)
    
    if len(mask_pixels) > 0:
        # Save image without mask
        if save_filepath is not None:
            fig.savefig(save_filepath + '_no_mask.png')
        
        # Create patches only for mask pixels (not all 262k pixels)
        for i, j in mask_pixels:
            patch_coords = [(j, i), (j, i+1), (j+1, i+1), (j+1, i)]
            patch = patches.Polygon(patch_coords, closed=True, fill=True, color='red', alpha=0.5)
            ax.add_patch(patch)
        
        # Save image with mask
        if save_filepath is not None:
            fig.savefig(save_filepath + '.png')
    
    plt.close(fig)
                
def draw_mask(img, mask):
    fig, ax = plt.subplots(1,2,figsize=(20,20))
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(img, cmap="gray")
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] > 0:
                patch_coords = [(j,i),(j,i+1),(j+1,i+1),(j+1,i)]
                patch = patches.Polygon(patch_coords, closed=True, fill=True, color='red', alpha=0.5)
                ax[1].add_patch(patch)
                #print('red')
                
                
def draw_pred_mask(img, pred, mask):
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(img, cmap="gray")
    ax[2].imshow(img, cmap="gray")
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] > 0:
                patch_coords = [(j,i),(j,i+1),(j+1,i+1),(j+1,i)]
                patch = patches.Polygon(patch_coords, closed=True, fill=True, color='red', alpha=0.5)
                ax[1].add_patch(patch)
                #print('red')
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i,j] > 0:
                patch_coords = [(j,i),(j,i+1),(j+1,i+1),(j+1,i)]
                patch = patches.Polygon(patch_coords, closed=True, fill=True, color='red', alpha=0.5)
                ax[2].add_patch(patch)
                #print('red')
    display(fig)
    plt.close()

def draw_first_positive(img_vol, pred_vol, mask_vol, batch_index):
    i = img_vol[batch_index,0,:,:,:].squeeze()
    p = pred_vol[batch_index,0,:,:,:].squeeze()
    t = mask_vol[batch_index,0,:,:,:].squeeze()
    #print(i.shape, p.shape, t.shape)
    a = 0 
    for iidx in range(0, t.shape[2]):
        maxval = torch.max(t[:,:,iidx])
        if maxval > 0.01:
            #print('test')
            if a == 1:
                #print('ping')
                #print(iidx, i.shape, p.shape)
                draw_pred_mask(i[:,:,iidx], p[:,:,iidx], t[:,:,iidx])
                break
            a += 1
    #print('a: %s' % a)

def draw_first_positive_2d_in_batch(img_vol, pred_vol, mask_vol):
    i = img_vol[:,0,:,:].squeeze()
    p = pred_vol[:,0,:,:].squeeze()
    t = mask_vol[:,0,:,:].squeeze()
    #print(i.shape, p.shape, t.shape)
    a = 0 
    for iidx in range(0, t.shape[0]):
        maxval = torch.max(t[iidx,:,:])
        if maxval > 0.01:
            #print('test')
            if a == 1:
                #print('ping')
                #print(iidx, i.shape, p.shape)
                draw_pred_mask(i[iidx,:,:], p[iidx,:,:], t[iidx,:,:])
                break
            a += 1
    #print('a: %s' % a)

def draw_volume_grid(volume, grid_dim, size=30):
    fig, ax = plt.subplots(grid_dim, grid_dim, figsize=(size, size))
    grid_count = grid_dim ** 2 
    for row in range(0, grid_dim):
      for col in range(0, grid_dim):
        slice = volume[:,:, int((volume.shape[2]/grid_count) * (row*grid_dim + col))]
        ax[row, col].imshow(slice, cmap='gray')

def show_png_grid(png_file_paths, size=30):
    cols = 4
    rows = math.ceil(len(png_file_paths)/cols)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * size, rows * size))
    for i, ax in enumerate(axs.flatten()):
        if i < len(mask_files):
            img = mpimg.imread(mask_files[i])
            ax.imshow(img) 
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def draw_all_pos_predicted_slices(img_vol, pred_vol):
    for z in range(0,pred_vol.shape[2]):
          slice_max = torch.max(pred_vol[:,:,z])
          if slice_max > 0.001:
              draw_mask(img_vol[:,:,z], pred_vol[:,:,z])

