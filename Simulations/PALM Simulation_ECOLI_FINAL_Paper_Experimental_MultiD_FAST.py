import cv2
import numpy as npy
import random as rnd
from scipy import ndimage
import math as mt
from skimage.external import tifffile
from scipy.stats import expon
import pandas as pd
import scipy.io as sio
#UNITs of time are in milliseconds
filename = 'Ecoli_Exp_MultiD_FAST_0.1s_10.tif'
pixels = 500
resolution = 10 #tens of nanometers
size_image = pixels*resolution # size of image in 10 nm resolution. Memory issues arise for better resolution
images = []
image = npy.zeros((size_image, size_image, 3), npy.uint16) # Create and RGB image
interval_frame = 1# Time interval (frames)
camera_exposure = 100
dt = 5#ms
int_steps = npy.int_(camera_exposure/dt)
num_images = 150

track_steps = int_steps*num_images*interval_frame
#setting the number of time steps based on exposure, time interval, number of images

print(track_steps)

integrate_array = npy.arange(int_steps - 1,track_steps, int_steps*interval_frame)
integrate_array_2 = npy.arange(int_steps - 1,track_steps, int_steps*interval_frame)
integrate_array_2[0] = 0
integrate_array_2[1:] = integrate_array_2[1:] - int_steps + 1
#print(integrate_array_2)
#integrate_array = npy.arange(int_steps - 1,track_steps, int_steps*interval_frame)
print(integrate_array)
#Array that tells when to snap an image
track_numbers = 2
#number of tracks per bacterium
num_bacteria = 400
#number of bacteria

int_fluorophore = 3000
#Intensity within exposure time
U1_single_spot = int_fluorophore/int_steps
#Intensity per single time step

out_background = 150
#Camera noise
out_back_sigma = 20
#standard deviation of camera noise
Cell_bg_intensity = 20
#Cell autofluorescence within exposure time
cell_background = 0.2#Cell_bg_intensity/int_steps
#cell_background_sigma = 0.05
mean_psf = 13
#Point spread function. I.e how much the intensity of the spot will be spread out based on convultion with gaussian filter
std_psf = 1.0 #don't use

wid_bac_mean = 70
#mean width of bacterium
std_wid = 10

length_bac_mean = 300
#mean length of bacterium
std_length = 50

Tbleach = 10000 # Bleach time (ms). Will need to vary this based on time interval
#kbleach = 1/Tbleach
Tunbind = 3000
Tunbind_2 = 15000
#Bound time
Tbind = 10000000000 # Time to bind (i.e. search time)
#kunbind = 1/Tunbind
#kbind = 1/Tbind
alpha_co = 0.4 # for chromosome bound fraction

D1 = 5# Diffusion coefficient of fast moving population
var_1 = 2*D1*dt

D2 = 0.05 # Diffusion coefficient of slow moving population in 10nm^2/mss
var_2 = 2*D2*dt**alpha_co

D3 = 0.05
var_3 = 2*D3*dt**alpha_co

#Weights of Diffusion coefficients
w1 = 0.1
w2 = 0.45
w3 = 1 - w1 - w2
params_array = [interval_frame, camera_exposure,cell_background, Tbleach,Tunbind,Tunbind_2, Tbind,D1,D2,  int_fluorophore, w1, w2]
params_array_num = npy.array(params_array)
dat_df = pd.DataFrame(params_array_num)
dat_df_Tr = dat_df.T
#print(dat_df_Tr)
columns_nam = ['Interval(frames)', 'Camera Exposure', 'Cell Background (per resolution)', 'Tbleach', 'Bound Time', 'Bound Time 2', 'Search Time', 'Fast Moving D', 'Bound D', 'IntFluorophore', 'Weight D1', 'Weight D2']
#print(columns_nam)
dat_df_Tr.columns = columns_nam



#df_params = pd.DataFrame(dat_df_Tr)#pd.DataFrame(dat_df_Tr, columns = columns_nam)

file_nam_params = filename.replace(".tif", "_params.csv")
dat_df_Tr.to_csv(file_nam_params)
#Array for positions of bacteria
bac_cells = npy.zeros((num_bacteria, 6))

#Borders of image
positions_lower_bnd = 0
positions_upper_bnd = 5000
#Array for initial positions of tracks
track_initials = npy.zeros((track_numbers, 3, num_bacteria))
#Array for tracks
track = npy.zeros((track_steps, 3, track_numbers, num_bacteria))
#Array for state
track_D = npy.zeros((track_numbers, 1, num_bacteria))


tracks_save = npy.zeros((num_images,2, track_numbers,num_bacteria))
tracks_save_D = npy.zeros((num_images,1, track_numbers,num_bacteria))

#States for Diffusive State
D1_st = 0
D2_st = 1
D3_st = 2
#States of fluorescence
FL_st = 3
Bl_st = 4
#Probabilities for diffusive state changes
D1_F_p = expon.cdf(x=dt, scale=Tbind)
D2_F_p = expon.cdf(x=dt, scale=Tunbind)
D3_F_p = expon.cdf(x=dt, scale=Tunbind_2)
#Bleach probability
B_p = expon.cdf(x=dt, scale=Tbleach)
#molecule can't switch to different bound states
D2_D3_p = 0
D3_D2_p = 0
#Probabilities for no transition
D1_D1_p = 1 - D1_F_p - D1_F_p #assuming equal probability of binding to either bound state (D2, D3)
D2_D2_p = 1 - D2_F_p - D2_D3_p
D3_D3_p = 1 - D3_F_p - D3_D2_p
B_p_N = 1 - B_p


Trans_mat_D = npy.array([[D1_D1_p, D1_F_p, D1_F_p], [D2_F_p, D2_D2_p, D2_D3_p], [D3_F_p, D3_D2_p, D3_D3_p]])
Trans_mat_F = npy.array([B_p_N, B_p])

weights = [w1, w2, w3]
D = [D1_st, D2_st, D3_st]

yrang_coord = npy.arange(positions_lower_bnd + (length_bac_mean + 3.5*std_length), positions_upper_bnd - (length_bac_mean + 3.5*std_length), (length_bac_mean + 3.5*std_length))
xrang_coord = npy.arange(positions_lower_bnd + (wid_bac_mean + 3.5*std_wid), positions_upper_bnd - (wid_bac_mean + 3.5*std_wid), (wid_bac_mean + 3.5*std_wid))

[xv, yv] = npy.meshgrid(xrang_coord, yrang_coord)
num_cols = npy.shape(xv)
print(num_cols)
coord_bac_x = []
coord_bac_y = []
for uu in range(num_cols[1]):
    for vv in range(num_cols[0] ):
        #x_coord_bac_sel = uu
        #y_coord_bac_sel = vv
        coord_bac_x.append(xv[0,uu])
        coord_bac_y.append(yv[vv,0])

coords_final = [coord_bac_x,coord_bac_y]
coords_final = [[row[i] for row in coords_final]
                             for i in range(len(coords_final[0]))]
coords_final = npy.array(coords_final)
#print(coords_final)

for u in range(num_bacteria):
    #Sampling for bacterial coordinates and dimensions
    x_coord_bac = coords_final[u,0]
    y_coord_bac = coords_final[u,1]
    wid_sam = int(rnd.normalvariate(wid_bac_mean, std_wid))
    length_sam = int(rnd.normalvariate(length_bac_mean, std_length))
    bac_cells[u, (0, 1)] = [x_coord_bac, y_coord_bac]  # coordinate of center of bacterial cell
    bac_cells[u, 2] = x_coord_bac + wid_sam
    bac_cells[u, 3] = y_coord_bac + length_sam
    bac_cells[u, 4] = wid_sam
    bac_cells[u, 5] = length_sam
    r = bac_cells[u, 4] / 2 # radius of bacterial cell

    xmin = bac_cells[u, 0]
    xmax = bac_cells[u, 2]
    ymin = bac_cells[u, 0]
    ymax = bac_cells[u, 2]
    zmin = bac_cells[u, 1]
    zmax = bac_cells[u, 3]

    for w in range(track_numbers):
        x_init = rnd.randrange(xmin, xmax)
        x_origin = (xmax - xmin) / 2 + xmin
        y_init = rnd.randrange(ymin, ymax)
        y_origin = (ymax - ymin) / 2 + ymin
        z_init = rnd.randrange(zmin, zmax)
        #Initial localizations for tracks
        track_initials[w, :, u] = [x_init, y_init, z_init]
        #Restricting localizations to within bacterial cells
        while (track_initials[w, 2, u] < zmin) or (track_initials[w, 2, u] > zmax) or (
                            (track_initials[w, 0, u] - x_origin) ** 2 +
                            (track_initials[w, 1, u] - y_origin) ** 2 > r ** 2):
            x_init = rnd.randrange(xmin, xmax)
            y_init = rnd.randrange(ymin, ymax)
            z_init = rnd.randrange(zmin, zmax)
            track_initials[w, :, u] = [x_init, y_init, z_init]
        track[0, :, w, u] = track_initials[w, :, u]
        #Selecting initial diffusion state
        Diffusion_selection = npy.random.multinomial(1, weights)
        Diffusion_selection_elem = npy.nonzero(Diffusion_selection)
        Diffuse_state = npy.sum(Diffusion_selection_elem)
        track_D [w, 0, u] = Diffuse_state

#Placing bacteria in image
for j in range(num_bacteria):
    bac_image = cv2.rectangle(image, (int(bac_cells[j, 0]), int(bac_cells[j, 1])), (int(bac_cells[j, 2]),
                                    int(bac_cells[j, 3])), (0, 255, 0), -1, 8)

bac_image_final = cv2.cvtColor(bac_image, cv2.COLOR_BGR2GRAY)

bac_image_16_reform = bac_image_final.reshape(pixels, resolution, pixels, resolution).sum(3).sum(1)
bac_image_16 = npy.uint16(bac_image_16_reform)
non_zer_find = npy.nonzero(bac_image_16)
bac_image_16[bac_image_16!=0] = 1
filename_bin = filename.replace(".tif", "_binary.tif")
#filename_bin.replace(".tif", "binary.tif")
#print(filename_bin)
cv2.imwrite(filename_bin, bac_image_16)
#Opening big tiff file
with tifffile.TiffWriter(filename, bigtiff=True) as tif:
    #Convert image to array
    bac_array = npy.array(bac_image_final)
    max_cell_value = npy.amax(bac_array)
    #Find pixels where bacteria cells are
    index_max = npy.argwhere(bac_array == max_cell_value)

    #Give cells autofluorescence
    for m in range(len(index_max)):
        bac_array[index_max[m, 0], index_max[m, 1]] = npy.random.poisson(cell_background, 1)
    bac_array_2 = npy.array(bac_array)

    for i in range(track_steps):
        if i in integrate_array_2:
            init_time = npy.where(integrate_array_2 == i)
            init_time = init_time[0]
            #print(init_time)
            int_init = integrate_array_2[init_time].item()
            int_last = integrate_array[init_time].item()
            #print(int_last)
            integrate_array_3 = npy.arange(int_init,int_last + 1)
            #print(integrate_array_3)
        for u in range(num_bacteria):
            r = bac_cells[u, 4] / 2
            xmin = bac_cells[u, 0]
            xmax = bac_cells[u, 2]
            ymin = bac_cells[u, 0]
            ymax = bac_cells[u, 2]
            zmin = bac_cells[u, 1]
            zmax = bac_cells[u, 3]
            x_origin = (xmax - xmin) / 2 + xmin
            y_origin = (ymax - ymin) / 2 + ymin

            for w in range(track_numbers):
                #Setup initial positions of tracks
                if i == 0:
                    spot_int = U1_single_spot
                    spot_coord = npy.zeros((1, 2))
                    spot_coord[0, (0, 1)] = [track[i, 0, w, u], track[i, 2, w, u]]
                    spot_coord = npy.int_(spot_coord)
                    gauss_sam = mean_psf#rnd.normalvariate(mean_psf, std_psf)
                    init_spot = npy.zeros((70, 70))
                    init_spot[34, 34] = spot_int

                    gauss_filter = ndimage.gaussian_filter(init_spot, gauss_sam, truncate=8)
                    for q in range(len(gauss_filter)):
                        for v in range(len(gauss_filter)):
                            poiss_dist = npy.random.poisson(gauss_filter[q, v], 1)
                            gauss_filter[q, v] = poiss_dist
                    spot_replace = bac_array_2[spot_coord[0, 1] - 34:spot_coord[0, 1] + 36,
                                   spot_coord[0, 0] - 34:spot_coord[0, 0] + 36]
                    if npy.size(spot_replace) != npy.size(gauss_filter):
                        print("Work on your coding skills you idiot!")
                    spot_replace_2 = npy.add(spot_replace, gauss_filter)
                    bac_array_2[spot_coord[0, 1] - 34:spot_coord[0, 1] + 36, spot_coord[0, 0] - 34:spot_coord[0, 0] + 36] = \
                        spot_replace_2
                    continue

                #If molecule is bleached we don't care about it
                if track_D [w, 0, u]  == Bl_st:
                    continue

                #Determine if molecule bleaches within next time step
                Transition_BL = Trans_mat_F
                rand_multi_select_BL = npy.random.multinomial(1, Transition_BL)
                element_multi_BL = npy.nonzero(rand_multi_select_BL)
                state_select_BL = npy.sum(element_multi_BL)

                if state_select_BL + 3 == FL_st:
                #If it is fluorescence determine if molecule transitions to different diffusive state
                    if track_D[w, 0, u] == D1_st:
                        Transition_p = Trans_mat_D[D1_st, :]
                        rand_multi_select = npy.random.multinomial(1, Transition_p)
                        element_multi = npy.nonzero(rand_multi_select)
                        state_select = npy.sum(element_multi)

                        if state_select == D1_st:
                            track_D[w, 0, u] = D1_st
                            var = var_1

                        elif state_select == D2_st:
                            track_D[w, 0, u] = D2_st
                            var = var_2

                        elif state_select == D3_st:
                            track_D[w, 0, u] = D3_st
                            var = var_3

                    elif track_D[w, 0, u] == D2_st:
                        Transition_p = Trans_mat_D[D2_st, :]
                        rand_multi_select = npy.random.multinomial(1, Transition_p)
                        element_multi = npy.nonzero(rand_multi_select)
                        state_select = npy.sum(element_multi)

                        if state_select == D1_st:
                            track_D[w, 0, u] = D1_st
                            var = var_1

                        elif state_select == D2_st:
                            track_D[w, 0, u] = D2_st
                            var = var_2

                        elif state_select == D3_st:
                            track_D[w, 0, u] = D3_st
                            var = var_3

                    elif track_D[w, 0, u] == D3_st:
                        Transition_p = Trans_mat_D[D3_st, :]
                        rand_multi_select = npy.random.multinomial(1, Transition_p)
                        element_multi = npy.nonzero(rand_multi_select)
                        state_select = npy.sum(element_multi)

                        if state_select == D1_st:
                            track_D[w, 0, u] = D1_st
                            var = var_1

                        elif state_select == D2_st:
                            track_D[w, 0, u] = D2_st
                            var = var_2

                        elif state_select == D3_st:
                            track_D[w, 0, u] = D3_st
                            var = var_3

                    #Pick a step size based on a normal distribution with variance determined by D value
                    x_step = rnd.normalvariate(0, mt.sqrt(var))
                    y_step = rnd.normalvariate(0, mt.sqrt(var))
                    z_step = rnd.normalvariate(0, mt.sqrt(var))
                    track[i, 0, w, u] = track[i - 1, 0, w, u] + x_step
                    track[i, 1, w, u] = track[i - 1, 1, w, u] + y_step
                    track[i, 2, w, u] = track[i - 1, 2, w, u] + z_step
                    #Make sure molecules are still within bacterial cell. Sample until it's within bounds
                    while (track[i, 2, w, u] < zmin) or (track[i, 2, w, u] > zmax) or ((track[i, 0, w, u] - x_origin) ** 2 +

                                                                        (track[i, 1, w, u] - y_origin) ** 2 > r ** 2):
                        x_step = rnd.normalvariate(0, mt.sqrt(var))
                        y_step = rnd.normalvariate(0, mt.sqrt(var))
                        z_step = rnd.normalvariate(0, mt.sqrt(var))
                        track[i, 0, w, u] = track[i - 1, 0, w, u] + x_step
                        track[i, 1, w, u] = track[i - 1, 1, w, u] + y_step
                        track[i, 2, w, u] = track[i - 1, 2, w, u] + z_step
                    #Spot intensity


                    if i in integrate_array_3:
                        #print(i, "Timestep Integrate 3")
                        spot_int = U1_single_spot

                        spot_coord = npy.zeros((1, 2))
                        spot_coord[0, (0, 1)] = [track[i, 0, w, u], track[i, 2, w, u]]
                        spot_coord = npy.int_(spot_coord)
                        #Sample a PSF value
                        gauss_sam = mean_psf#rnd.normalvariate(mean_psf, std_psf)
                        init_spot = npy.zeros((70, 70))
                        #Put Integrated intensity into center of array
                        init_spot[34, 34] = spot_int

                        #Add gaussian filter to intensity to spread out intensity
                        gauss_filter = ndimage.gaussian_filter(init_spot, gauss_sam, truncate=8)

                        #For each coordinate, sample based on Poisson Shot noise
                        for q in range(len(gauss_filter)):
                            for v in range(len(gauss_filter)):
                                poiss_dist = npy.random.poisson(gauss_filter[q, v], 1)
                                gauss_filter[q, v] = poiss_dist
                        spot_replace = bac_array_2[spot_coord[0, 1] - 34:spot_coord[0, 1] + 36,
                                                 spot_coord[0, 0] - 34:spot_coord[0, 0] + 36]

                        if npy.size(spot_replace) != npy.size(gauss_filter):
                            print("Work on your coding skills you idiot!")

                        #Add spot intensity to intensities at original location
                        spot_replace_2 = npy.add(spot_replace, gauss_filter)
                        #Replace original coordinates with revised intensity at that region
                        bac_array_2[spot_coord[0, 1] - 34:spot_coord[0, 1] + 36, spot_coord[0, 0] - 34:spot_coord[0, 0] + 36] = \
                            spot_replace_2
                #If bleach state is picked continue
                elif state_select_BL + 3 == Bl_st:
                    track_D[w, 0, u] = Bl_st
                    continue
        print (i, 'TimeStep')
        #Save image if it's integrate array. Based on exposure time and time interval
        if i in integrate_array:
            print (i, "save I")
            #Convert to 16-bit array
            bac_array_int16 = npy.uint16(bac_array_2)
            #Reformat for pixels
            bac_array_reform = bac_array_int16.reshape(pixels, resolution, pixels, resolution).sum(3).sum(1)
            #Add EMCCD camera noise
            for m in range(len(bac_array_reform)):
                for n in range(len(bac_array_reform)):
                    bac_array_reform[m, n] = bac_array_reform[m, n] + npy.random.normal(out_background, out_back_sigma, 1)

            tif.save(npy.uint16(bac_array_reform))


            bac_array = npy.array(bac_image_final)
            max_cell_value = npy.amax(bac_array)
            index_max = npy.argwhere(bac_array == max_cell_value)
            for m in range(len(index_max)):
                bac_array[index_max[m, 0], index_max[m, 1]] = npy.random.poisson(cell_background, 1)


            bac_array_2 = npy.array(bac_array)

            time_pt = npy.where(integrate_array == i)
            #print(time_pt)
            time_pt = time_pt[0]
            #print(time_pt)


            if time_pt == 0:
                time_step_prev = 0
            else:
                time_pt_prev = time_pt - 1
                time_step_prev = integrate_array[time_pt_prev].item()
            time_step_curr = integrate_array[time_pt].item()
            #print(integrate_array[time_pt_prev],integrate_array[time_pt])
            print(time_step_prev, time_step_curr)
            for u in range(num_bacteria):
                for w in range(track_numbers):

                    x_mean_int = npy.mean(track[time_step_prev:time_step_curr, 0, w, u])
                    y_mean_int = npy.mean(track[time_step_prev:time_step_curr, 2, w, u])
                    tracks_save[time_pt, 0, w, u] = x_mean_int
                    tracks_save[time_pt, 1, w, u] = y_mean_int
                    tracks_save_D[time_pt, 0, w, u] = track_D[w, 0, u]
            print('done')
file_nam_tracks = filename.replace(".tif", "_track_results.mat")
file_nam_tracks_D = filename.replace(".tif", "_track_results_D.mat")
sio.savemat(file_nam_tracks,{'tracks_info' : tracks_save})
sio.savemat(file_nam_tracks_D,{'tracks_info_D' : tracks_save_D})






