%function [multiscale_basis,quadrant]=multiscale_cal(global_stiffness_matrix, total_num_fine, num_coarse, num_fine);

num_fine=4;
num_coarse=2;
local_fine_scale_stiffness_matrices=zeros(num_fine-1, num_fine-1, num_coarse*num_coarse);