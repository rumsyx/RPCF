function params=testing()

close all

% s_frames = seq.s_frames;

% Feature specific parameters
hog_params.cell_size = 4;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 1;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;

ic_params.tablename = 'intensityChannelNorm6';
ic_params.useForColor = false;
ic_params.cell_size = 4;

cnn_params.nn_name = 'imagenet-vgg-m-2048.mat'; % Name of the network
cnn_params.output_layer = [0 3 14];             % Which layers to use
cnn_params.downsample_factor = [4 2 1];         % How much to downsample each output layer
% params.compressed_dim = [3 16 64 10 3];
% params.compressed_dim = [3 96 512 31 10]; 
params.compressed_dim = [3 36 129 10 6];  
cnn_params.input_size_mode = 'adaptive';        % How to choose the sample size
cnn_params.input_size_scale = 1;                % Extra scale factor of the input samples to the network (1 is no scaling)

% Which features to include
params.t_features = {
    struct('getFeature',@get_cnn_layers, 'fparams',cnn_params),...
      struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...
    ...struct('getFeature',@get_table_feature, 'fparams',ic_params),...
};

% Global feature parameters
params.t_global.normalize_power = 2;    % Lp normalization with this p
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature
params.t_global.normalize_dim = true;   % Also normalize with respect to the dimensionality of the feature

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale =4.5;         % The scaling of the target size to get the search area
% params.search_area_scale = 3.0;   
params.min_image_sample_size = 210^2;   % Minimum area of image samples
params.max_image_sample_size = 300^2;   % Maximum area of image samples

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score

% Learning parameters
params.output_sigma_factor = 1/12;		% Label function sigma
params.learning_rate = 0.025;		% Learning rate
params.nSamples = 50;                  % Maximum number of stored training samples
params.sample_replace_strategy = 'lowest_prior';    % Which sample to replace when the memory is full
params.lt_size = 0;                     % The size of the long-term memory (where all samples have equal weight)

% Conjugate Gradient parameters
params.max_CG_iter = 15;                 % The number of Conjugate Gradient iterations
params.init_max_CG_iter = 100;          % The number of Conjugate Gradient iterations used in the first frame
params.CG_tol = 1e-3;                   % The tolerence of CG does not have any effect
params.CG_forgetting_rate = 10;         % Forgetting rate of the last conjugate direction
params.precond_data_param = 0.5;        % Weight of the data term in the preconditioner
params.precond_reg_param = 0.01;        % Weight of the regularization term in the preconditioner


params.CG_forgetting_rate = 75;         % Forgetting rate of the last conjugate direction
params.learning_rate = 0.02;		% Learning rate
%% 
params.neglect_higher_frequency = false; 
params.use_sample_merge = true; 
params.sample_update_criteria = 'Merge';        % Strategy for updating the samples
params.weight_update_criteria = 'WeightedAdd';  % Strategy for updating the distance matrix
params.train_gap = 6;                   % The number of intermediate frames with no training (0 corresponds to training every frame)
params.skip_after_frame = 1;            % After which frame number the sparse update scheme should start (1 is directly)

% Regularization window parameters
params.use_reg_window = true;           % Use spatial regularization or not
params.reg_window_min = 1e-4;			% The minimum value of the regularization window
params.reg_window_edge = 10e-3;         % The impact of the spatial regularization
params.reg_window_power = 2;            % The degree of the polynomial to use (e.g. 2 is a quadratic window)
params.reg_sparsity_threshold = 0.05;   % A relative threshold of which DFT coefficients that should be set to zero

% Interpolation parameters
params.interpolation_method = 'bicubic';    % The kind of interpolation kernel
params.interpolation_bicubic_a = -0.75;     % The parameter for the bicubic interpolation kernel
params.interpolation_centering = true;      % Center the kernel at the feature sample
params.interpolation_windowing = false;     % Do additional windowing on the Fourier coefficients of the kernel

% Scale parameters
params.number_of_scales = 5;            % Number of scales to run the detector
params.scale_step = 1.02;               % The scale factor

% Other parameters
params.visualization = 1;               % Visualiza tracking and detection scores
params.debug = 0;                       % Do full debug visualization
params.frag_num=9;

% Initialize
% params.wsize = [seq.init_rect(1,4), seq.init_rect(1,3)];
% params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
% params.s_frames = s_frames;

% Run tracker
% results = tracker(params);
