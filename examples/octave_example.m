%% vision-calib Octave/MATLAB Example
%% This script demonstrates how to read calibration data exported by vision-calib
%%
%% Supported formats:
%%   - HDF5 (.h5)  : Use h5read() function
%%   - MAT  (.mat) : Use load() function (native)
%%
%% Author: vision-calib contributors
%% License: Apache 2.0

clear all; close all; clc;

%% ========================================
%% Method 1: Load from MAT file (Recommended for Octave)
%% ========================================

fprintf('=== Loading from MAT file ===\n');

% Load MAT file - all variables loaded into struct
if exist('calibration.mat', 'file')
    data = load('calibration.mat');

    % Camera intrinsic matrix (3x3)
    K = data.camera_matrix;
    fprintf('Camera Matrix K:\n');
    disp(K);

    % Distortion coefficients [k1, k2, p1, p2, k3, ...]
    D = data.distortion_coeffs;
    fprintf('Distortion Coefficients D:\n');
    disp(D);

    % Image size [width, height]
    image_size = data.image_size;
    fprintf('Image Size: %d x %d\n', image_size(1), image_size(2));

    % Reprojection error
    rms_error = data.reprojection_error;
    fprintf('Reprojection Error: %.4f pixels\n', rms_error);

    % Extract camera parameters
    fx = K(1,1);  % Focal length X
    fy = K(2,2);  % Focal length Y
    cx = K(1,3);  % Principal point X
    cy = K(2,3);  % Principal point Y

    fprintf('\nCamera Parameters:\n');
    fprintf('  fx = %.2f pixels\n', fx);
    fprintf('  fy = %.2f pixels\n', fy);
    fprintf('  cx = %.2f pixels\n', cx);
    fprintf('  cy = %.2f pixels\n', cy);

    % Check for extrinsic parameters
    if isfield(data, 'rotation_vector')
        fprintf('\nExtrinsic Parameters:\n');
        fprintf('  Rotation Vector: [%.6f, %.6f, %.6f]\n', data.rotation_vector);
        fprintf('  Translation Vector: [%.2f, %.2f, %.2f]\n', data.translation_vector);
    end

    % Check for checkerboard info
    if isfield(data, 'checkerboard_size')
        fprintf('\nCheckerboard: %d x %d, square size = %.1f mm\n', ...
            data.checkerboard_size(1), data.checkerboard_size(2), ...
            data.square_size_mm);
    end
else
    fprintf('calibration.mat not found. Skipping MAT example.\n');
end

%% ========================================
%% Method 2: Load from HDF5 file
%% ========================================

fprintf('\n=== Loading from HDF5 file ===\n');

if exist('calibration.h5', 'file')
    % Read intrinsic parameters
    K_h5 = h5read('calibration.h5', '/intrinsic/camera_matrix');
    D_h5 = h5read('calibration.h5', '/intrinsic/distortion_coeffs');
    image_size_h5 = h5read('calibration.h5', '/intrinsic/image_size');

    fprintf('Camera Matrix K (from HDF5):\n');
    disp(K_h5);

    % Read metadata
    info = h5info('calibration.h5');
    fprintf('HDF5 Groups: ');
    for i = 1:length(info.Groups)
        fprintf('%s ', info.Groups(i).Name);
    end
    fprintf('\n');

    % Read extrinsic if available
    try
        rvec = h5read('calibration.h5', '/extrinsic/rotation_vector');
        tvec = h5read('calibration.h5', '/extrinsic/translation_vector');
        fprintf('Extrinsic parameters found in HDF5.\n');
    catch
        fprintf('No extrinsic parameters in HDF5 file.\n');
    end
else
    fprintf('calibration.h5 not found. Skipping HDF5 example.\n');
end

%% ========================================
%% Example: Undistort a point
%% ========================================

fprintf('\n=== Example: Undistort Point ===\n');

if exist('K', 'var') && exist('D', 'var')
    % Example distorted point (pixel coordinates)
    distorted_point = [500; 400; 1];  % Homogeneous coordinates

    % Normalized coordinates
    normalized = K \ distorted_point;
    x = normalized(1);
    y = normalized(2);

    % Apply distortion model (simplified, radial only)
    r2 = x^2 + y^2;
    k1 = D(1);
    k2 = D(2);

    % Radial distortion factor
    radial_factor = 1 + k1*r2 + k2*r2^2;

    fprintf('Original point: (%.1f, %.1f)\n', distorted_point(1), distorted_point(2));
    fprintf('Normalized: (%.4f, %.4f)\n', x, y);
    fprintf('Radial factor: %.6f\n', radial_factor);
end

%% ========================================
%% Helper function: Convert rotation vector to matrix
%% ========================================

function R = rodrigues(rvec)
    % Convert rotation vector to rotation matrix (Rodrigues formula)
    theta = norm(rvec);
    if theta < 1e-10
        R = eye(3);
        return;
    end

    k = rvec / theta;  % Unit vector
    K = [0, -k(3), k(2); k(3), 0, -k(1); -k(2), k(1), 0];  % Skew-symmetric
    R = eye(3) + sin(theta)*K + (1-cos(theta))*K*K;
end

fprintf('\n=== Done ===\n');
