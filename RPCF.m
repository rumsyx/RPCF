function results = RPCF(seq)

% command=['/usr/local/cuda-8.0/bin/nvcc -ptx my_fun.cu'];
% system(command);
% kernel = parallel.gpu.CUDAKernel( 'my_fun.ptx',  'my_fun.cu' );
% [a. b]=feval(kernel,a,b);
global cnnd;
global cnna;
global cnn_third;
jump=0;
accu_jump=0;
response_backup=0;
low_confidence=0;
cleanupObj = onCleanup(@cleanupFun);
% net = load(['networks/imagenet-vgg-m-2048.mat']);
% net = vl_simplenn_move(net, 'gpu');
rand('state', 0);
a=1;
gpuArray(a);
set_tracker_param;
params=testing();
%% read images
num_z = 4;
% im1_name = sprintf([data_path 'img/%0' num2str(num_z) 'd.jpg'], im1_id);
im = imread(seq.s_frames{1});
CG_tol = params.CG_tol;

params.wsize = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
pos = floor(params.init_pos(:)');
target_sz = floor(params.wsize(:)');
init_target_sz = target_sz;
search_area_scale = params.search_area_scale;
max_image_sample_size = params.max_image_sample_size;
max_image_sample_size = params.max_image_sample_size;
min_image_sample_size = params.min_image_sample_size;
refinement_iterations = params.refinement_iterations;
nScales = params.number_of_scales;
scale_step = params.scale_step;
features = params.t_features;

prior_weights = [];
sample_weights = [];
latest_ind = [];


if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
if ~isfield(params, 'interpolation_method')
    params.interpolation_method = 'none';
end
if ~isfield(params, 'interpolation_centering')
    params.interpolation_centering = false;
end
if ~isfield(params, 'interpolation_windowing')
    params.interpolation_windowing = false;
end
if ~isfield(params, 'clamp_position')
    params.clamp_position = false;
end

params.data_type = zeros(1, 'single', 'gpuArray');
params.data_type_complex = complex(params.data_type);
params.sample_merge_type = 'Merge';
%% new parameters for the gpu support
distance_matrix = inf(params.nSamples, 'single');
gram_matrix = inf(params.nSamples, 'single');

if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end
search_area = prod(init_target_sz * search_area_scale);
if search_area > max_image_sample_size
    currentScaleFactor = sqrt(search_area / max_image_sample_size);
elseif search_area < min_image_sample_size
    currentScaleFactor = sqrt(search_area / min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

%window size, taking padding into account
base_target_sz = target_sz / currentScaleFactor;
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*4]; % for testing
end
% img_sample_sz=[220,220];
% img_sample_sz(1)=max(img_sample_sz(1),207);
% img_sample_sz(2)=max(img_sample_sz(2),207);
[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');
features{1}.fparams.output_layer=[0 3 3];
features{1}.fparams.net.layers=features{1}.fparams.net.layers(1:3);
% features{1}.fparams
img_sample_sz = feature_info.img_sample_sz;
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;


caffe.reset_all();
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
feature_solver_def_file = 'model/feature_solver.prototxt';
model_file = 'model/VGG_ILSVRC_16_layers.caffemodel';

fsolver = caffe.Solver(feature_solver_def_file);
fsolver.net.copy_from(model_file);
feature_input = fsolver.net.blobs('data');
feature_blob4 = fsolver.net.blobs('conv4_3');

% feature_size=round([275 275]*sqrt(search_area)/400);
% feature_size=max(feature_size,[85 85]);
feature_size=[271 271];
tmp=extract_vgg16(zeros(feature_size(1),feature_size(2),3),fsolver,feature_input,feature_blob4,global_fparams);

feature_sz(3,:)=[size(tmp,1) size(tmp,2)];



feature_dim = feature_info.dim;
feature_dim=params.compressed_dim(1:length(feature_info.dim))';

reg_window_edge = {};
for k = 1:length(features)
    if isfield(features{k}.fparams, 'reg_window_edge')
        reg_window_edge = cat(3, reg_window_edge, permute(num2cell(features{k}.fparams.reg_window_edge(:)), [2 3 1]));
    else
        reg_window_edge = cat(3, reg_window_edge, cell(1, 1, length(features{k}.fparams.nDim)));
    end
end
[reg_filter, binary_mask, patch_mask, reg_window] = cellfun(@(reg_window_edge) get_reg_filter(img_support_sz, base_target_sz, params, reg_window_edge), reg_window_edge, 'uniformoutput', false);

num_feature_blocks = length(feature_dim);
% Size of the extracted feature maps

feature_reg = permute(num2cell(feature_info.penalty), [2 3 1]);
filter_sz = feature_sz; 

feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

set_tracker_param1



[x,x1,y,y1]=generate_displacement();
feature_input.dt_pooling_parameter(filter_sz(1,1),filter_sz(1,2),x,x1,y,y1);


for block_id=1:num_feature_blocks
      binary_mask{block_id}=imresize(binary_mask{block_id}, [filter_sz(block_id,1),filter_sz(block_id,2)], 'nearest');
      reg_window{block_id}=imresize(reg_window{block_id}, [filter_sz(block_id,1),filter_sz(block_id,2)], 'nearest');
end

[patch_mask]=get_binary_patch_mask(binary_mask,params.frag_num);
output_sz = max(filter_sz, [], 1); %% 
pad_sz = cellfun(@(filter_sz) (output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false); %% 


ky = circshift(-floor((output_sz(1) - 1)/2) : ceil((output_sz(1) - 1)/2), [1, -floor((output_sz(1) - 1)/2)])';
kx = circshift(-floor((output_sz(2) - 1)/2) : ceil((output_sz(2) - 1)/2), [1, -floor((output_sz(2) - 1)/2)]);
ky_tp = ky';
kx_tp = kx';

cos_window = cellfun(@(sz) single(hann(sz(1)+2)*hann(sz(2)+2)'), feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cos_window(2:end-1,2:end-1), cos_window, 'uniformoutput', false);

[interp1_fs, interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

%% considers scale
if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    
    scaleFactors = scale_step .^ scale_exp;
    
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

%% %% extract features
samplesf = cell(1, 1, num_feature_blocks);
for k = 1:num_feature_blocks
    samplesf{k} = complex(zeros(params.nSamples,feature_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2,'single'));
end

for block_id=1:num_feature_blocks
    output_sigma_factor=0.1;
    cell_size=img_sample_sz./filter_sz(block_id,:);
    cell_size=mean(cell_size);
    output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;

end

%% construct y in fourier domain (yf)
sig_y = sqrt(prod(floor(base_target_sz))) * output_sigma_factor * (output_sz ./ img_support_sz);
yf_y = single(sqrt(2*pi) * sig_y(1) / output_sz(1) * exp(-2 * (pi * sig_y(1) * ky / output_sz(1)).^2));
yf_x = single(sqrt(2*pi) * sig_y(2) / output_sz(2) * exp(-2 * (pi * sig_y(2) * kx / output_sz(2)).^2));
y_dft = yf_y * yf_x*1;
yf = cellfun(@(sz) fftshift(resizeDFT2(y_dft, sz, false)), filter_sz_cell, 'uniformoutput', false);
  for block_id=1:num_feature_blocks
        y_real{1,1,block_id}=real(ifft2( ifftshift(ifftshift(yf {1,1,block_id},1),2) ));
     
  end

yf = compact_fourier_coeff(yf);
max_response1=max(max(y_real{1,1,1}));


params.nSamples = min(params.nSamples, numel(seq.s_frames));
for frame=1:seq.endFrame-seq.startFrame+1
       im = imread(seq.s_frames{frame});
       
       if frame>1
                if size(im,3) > 1 && is_color_image == false
                   im = im(:,:,1);
                end
%                 tic;
                xt = extract_features(im, pos, currentScaleFactor*scaleFactors, features, global_fparams,binary_mask,patch_mask);
%                 toc;            
                  for scale_ind=1:5
                      img_samples(:,:,:,scale_ind) = single(sample_patch(im, pos, round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind)), [feature_size(1) feature_size(2)]));
%                       img_samples(:,:,:,scale_ind) = single(sample_patch(im, pos, round(img_support_sz*currentScaleFactor), [feature_size feature_size]));
                     end
                    
                    img_samples = impreprocess(img_samples);
                   
                    [xt{3}, ignore]=extract_vgg16(img_samples,fsolver,feature_input,feature_blob4,global_fparams);
                    
%                     xt{2}=imresize(xt{2}, feature_sz(2,:), 'bilinear', 'Antialiasing',false);
                    
                     for block_id=2:num_feature_blocks
                       xt{block_id}=gpuArray(xt{block_id});
                     end
                    clear img_samples;
   
                
                xt = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
                 xt = project_sample(xt, projection_matrix,x_mean);
                
                xtf = cellfun(@cfft2, xt, 'uniformoutput', false);
                
                xtf = interpolate_dft(xtf, interp1_fs, interp2_fs);
                
                
             %% track the current frame using trained filter
             for block_id=1:num_feature_blocks
                      tmp=weight_f{block_id};
                     score_fs{1,1,block_id}=bsxfun(@times,conj(tmp), ((xtf{block_id})));
                     
                end
                
                scores_fs_feat = cellfun(@(score_fs, pad_sz) padarray(sum(score_fs, 3), pad_sz), score_fs, pad_sz, 'uniformoutput', false);
                scores_fs=scores_fs_feat{1};
                for block_id=2:num_feature_blocks
                    scores_fs=scores_fs+scores_fs_feat{block_id};
                end

                
                scores_fs=permute(scores_fs,[1 2 4 3]);
                scores_fs=ifftshift(ifftshift(scores_fs,1),2);
                newton_iterations=5;
                
               response= ifft2(scores_fs);
               response=real(fftshift(fftshift(response,1),2));

                [trans_row, trans_col, scale_ind] = optimize_scores(gather(scores_fs), newton_iterations, ky_tp, kx_tp);
                
                
                 translation_vec = round([trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(scale_ind));
                if (abs(trans_row(1))>filter_sz(1,1)/2*0.5||abs(trans_col(1))>filter_sz(1,2)/2*0.5 )&&(abs(translation_vec(1))>60||abs(translation_vec(2))>60)&&frame>20
                      translation_vec= translation_vec*0;
                end
%         
              
                
                if scale_ind~=3
                    a=1;
                end
            % set the scale
            currentScaleFactor = currentScaleFactor * scaleFactors(scale_ind);
            % adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            target_sz = floor(base_target_sz * currentScaleFactor);
            % update position
            old_pos = pos;
            pos = pos + translation_vec;
            
               if frame<10
                   params.learning_rate=0.011;
               else
                  params.learning_rate=0.02;
               end
              
%                if max(response(:))<0.25*max_response1&&frame>10
                   params.max_CG_iter = 3;
%                end
            
%                  pos = pos - translation_vec;
                visualization=1;
                rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
     if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end

        
        if frame == 2,  %first frame, create GUI
            figure('Name','Tracking Results');
% %             figure(1)
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            
            rect_handle = rectangle('Position', rect_position_vis, 'EdgeColor','r', 'linewidth', 2);
            text_handle = text(10, 10, sprintf('#%d / %d',seq.startFrame+frame-1, seq.endFrame));
            set(text_handle, 'color', [1 1 0], 'fontsize', 16, 'fontweight', 'bold');
%              imwrite(frame2im(getframe(gcf)),sprintf('result/%04d.bmp',frame));
        else
            set(im_handle, 'CData', uint8(im))
            set(rect_handle, 'Position', rect_position_vis)
            set(text_handle, 'string', sprintf('#%d / %d',seq.startFrame+frame-1, seq.endFrame));
%                imwrite(frame2im(getframe(gcf)),sprintf('result/%04d.bmp',frame));
        end
        
        drawnow
      end   
             results(frame,:)=rect_position_vis;    
                
       end
               %% extract features of training samples
                if size(im,3) > 1 && is_color_image == false
                   im = im(:,:,1);
                end
%                 tic;
                xl = extract_features(im, pos, currentScaleFactor, features, global_fparams,binary_mask,patch_mask);
%                    toc;
              for scale_ind=1:1
                 img_samples(:,:,:,scale_ind) = single(sample_patch(im, pos, round(img_support_sz*currentScaleFactor), [feature_size(1) feature_size(2)]));
              end
              
              img_samples = impreprocess(img_samples);
              [xl{3},ignore]=extract_vgg16(img_samples,fsolver,feature_input,feature_blob4,global_fparams);
              
              
              for block_id=3:num_feature_blocks
                   xl{block_id}=gpuArray(xl{block_id});
              end
          
               clear img_samples;
               
               
                
                xl = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
                
                if frame==1
                 feature_reduction;
               end
                %% Dimension reduction
                xl = project_sample(xl, projection_matrix,x_mean);
                
                xlf = cellfun(@cfft2, xl, 'uniformoutput', false);
                
                xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
                
                xlf = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:), xlf, 'uniformoutput', false);
                
            %% model update                         
               model_update;
                               
                 end
                
               feature_input.clear_memory_function(1);
               cnna.net.forward_prefilled();
               cnn_third.net.forward_prefilled();
               cnnb.net.forward_prefilled();
               













