rand('state',0);
% filter_sz(3,:)=[30 30];
if frame==1
        %% fft
       index_tmp=1:filter_sz(3,1)*(filter_sz(3,2)+1)/2;
       index_tmp=reshape(index_tmp,[(filter_sz(3,1)+1)/2, filter_sz(3,2)]);
       index_tmp=permute(index_tmp,[2 1 3 4]);
       index_tmp=padarray(index_tmp,pad_sz{3});
       L_index=index_tmp(:,1:(filter_sz(1,1)+1)/2);
       L_index=permute(L_index,[2 1 3 4]);
%        test1=permute(L_index,[2 1 3 4]);
       L_index1=find(L_index(:));
       L_index1=reshape(L_index1,[(filter_sz(3,1)+1)/2, filter_sz(3,2)]);
       feature_input.set_L_index(L_index,L_index1);
        num_training_samples=0;
       frames_since_last_train = inf;
        max_train_samples=params.nSamples;
        prior_weights = zeros(max_train_samples,1, 'single');
        params.minimum_sample_weight = params.learning_rate*(1-params.learning_rate)^(2*max_train_samples);
        score_matrix = inf(max_train_samples, 'single');
        latest_ind = [];
        samplesf = cell(1, 1, num_feature_blocks);
%         prior_weights = [];
        sample_weights = prior_weights;
        
        val_index(1)=0; 
        val_index(2)=filter_sz(1,1)*(filter_sz(1,2)+1)/2*(params.frag_num+1)*params.compressed_dim(1);
        val_index(4)=val_index(2)+filter_sz(1,1)*(filter_sz(1,2)+1)/2*(params.frag_num+1)*(params.compressed_dim(2));
        if num_feature_blocks>4
            val_index(5)=val_index(4)+filter_sz(1,1)*(filter_sz(1,2)+1)/2*(params.frag_num+1)*(params.compressed_dim(4));
        end
        val_index(3)=0;
        
        val_index1(1)=0; 
        val_index1(2)=filter_sz(1,1)*filter_sz(1,2)*(params.frag_num+1)*params.compressed_dim(1);
        val_index1(4)=val_index1(2)+filter_sz(1,1)*filter_sz(1,2)*(params.frag_num+1)*(params.compressed_dim(2));
        if num_feature_blocks>4
            val_index1(5)=val_index1(4)+filter_sz(1,1)*filter_sz(1,2)*(params.frag_num+1)*(params.compressed_dim(4));
        end
        val_index1(3)=0;
        total_num=filter_sz(:,1).*filter_sz(:,2).*feature_dim;
        total_num1=filter_sz(:,1).*(filter_sz(:,2)+1)/2.*feature_dim;
        total_num(3,:)=0;
        total_num1(3,:)=0;
        total_num=sum(total_num);
        total_num1=sum(total_num1);
        total_num2=filter_sz(3,1).*filter_sz(3,2).*feature_dim(3);
        total_num3=filter_sz(3,1).*(filter_sz(3,2)+1)/2.*feature_dim(3);
        
        
        feature_input.set_index(val_index',val_index1',total_num,total_num1,total_num2,total_num3);
        for k=1:num_feature_blocks
               feature_input.inialize_blobs(k,params.nSamples,params.compressed_dim(k),filter_sz(k,1),filter_sz(k,2),params.frag_num);
             samplesf{k} = zeros(filter_sz(k,1),(filter_sz(k,2)+1)/2,params.compressed_dim(k),params.nSamples, 'like', params.data_type_complex);
             samplesf{k}=single(samplesf{k});
        end
        feature_input.set_feature_num(num_feature_blocks);
        feature_input.input_yf(real(permute(yf{1},[2,1,3,4])), imag(permute(yf{1},[2,1,3,4])),...
             real(permute(yf{3},[2,1,3,4])), imag(permute(yf{3},[2,1,3,4])));
       feature_input.clear_memory_function(0);
    
fftshift_mask1=1:filter_sz(1,1)*( floor( filter_sz(1,2)/2)+1);
fftshift_mask1=reshape(fftshift_mask1,[floor( filter_sz(1,2)/2)+1  filter_sz(1,1) ])-1;
fftshift_mask1=permute(fftshift_mask1,[2 1]);
fftshift_mask1=process_freq(fftshift_mask1,filter_sz(1,1),filter_sz(1,2));
fftshift_mask1=fftshift(fftshift(fftshift_mask1,1),2);
fftshift_mask1=fftshift_mask1(:, 1:floor(filter_sz(1,2)/2)+1 );  
% 
fftshift_mask2=1:filter_sz(3,1)*( floor( filter_sz(3,2)/2)+1);
fftshift_mask2=reshape(fftshift_mask2,[floor( filter_sz(3,2)/2)+1  filter_sz(3,1) ])-1;
fftshift_mask2=permute(fftshift_mask2,[2 1]);
fftshift_mask2=process_freq(fftshift_mask2,filter_sz(3,1),filter_sz(3,2));
fftshift_mask2=fftshift(fftshift(fftshift_mask2,1),2);
fftshift_mask2=fftshift_mask2(:, 1:floor(filter_sz(3,2)/2)+1 );    
% 
%     
    ifftshift_mask1=1:filter_sz(1,1)*( floor( filter_sz(1,2)/2)+1); 
    ifftshift_mask1=reshape(ifftshift_mask1,[floor( filter_sz(1,2)/2)+1  filter_sz(1,1) ] );
    ifftshift_mask1=permute(ifftshift_mask1,[2 1]);
    ifftshift_mask1= full_fourier_coeff_ifftshift(ifftshift_mask1);
    ifftshift_mask1=ifftshift(ifftshift(ifftshift_mask1,1),2);
    ifftshift_mask1=ifftshift_mask1(:, 1:floor(filter_sz(1,2)/2)+1 );      
%     
    ifftshift_mask2=1:filter_sz(3,1)*( floor( filter_sz(3,2)/2)+1); 
    ifftshift_mask2=reshape(ifftshift_mask2,[floor( filter_sz(3,2)/2)+1  filter_sz(3,1) ] );
    ifftshift_mask2=permute(ifftshift_mask2,[2 1]);
    ifftshift_mask2= full_fourier_coeff_ifftshift(ifftshift_mask2);
    ifftshift_mask2=ifftshift(ifftshift(ifftshift_mask2,1),2);
    ifftshift_mask2=ifftshift_mask2(:, 1:floor(filter_sz(3,2)/2)+1 );    
% 
    fftshift_mask1=permute(fftshift_mask1,[2 1]);
    fftshift_mask2=permute(fftshift_mask2,[2 1]);
    ifftshift_mask1=permute(ifftshift_mask1,[2 1]);
    ifftshift_mask2=permute(ifftshift_mask2,[2 1]);
    
    feature_input.set_reg_window(fftshift_mask1,fftshift_mask2,ifftshift_mask1,ifftshift_mask2, ...
        permute(binary_mask{1},[2 1 3 4]),permute(binary_mask{3},[2 1 3 4]),...
        permute(reg_window{1},[2 1 3 4]),permute(reg_window{3},[2 1 3 4]));
    feature_input.set_patch_mask(  permute(patch_mask{1},[2 1 3 4]) , permute(patch_mask{3},[2 1 3 4])    );
    feature_input.set_binary_mask_adaptive(permute(binary_mask{1},[2 1 3 4]),permute(binary_mask{3},[2 1 3 4]));
% yf=repmat(yf,[1,1,1,params.nSamples]);
end
for block_id=1:num_feature_blocks
       fft_input{block_id}=fft2(single((xl{block_id})));
end

xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf, 'uniformoutput', false);

%% create & update sample space
[merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights] = ...
                 update_sample_space_model_gpu(samplesf, xlf, distance_matrix, gram_matrix, prior_weights,...
                 num_training_samples,params);
             
if num_training_samples < params.nSamples
        num_training_samples = num_training_samples + 1;
end             
             
sample_weights = prior_weights;
  feature_input.input_sample_weight(sample_weights);
  
for k = 1:num_feature_blocks
      if merged_sample_id > 0
                samplesf{k}(:,:,:,merged_sample_id) = merged_sample{k};
                feature_input.update_samplesf(gather(real(permute(merged_sample{k},[2 1 3 4]))),gather(imag(permute(merged_sample{k},[2 1 3 4]))),k,merged_sample_id);
            end
            
            if new_sample_id > 0
                samplesf{k}(:,:,:,new_sample_id) = new_sample{k};
                feature_input.update_samplesf(gather(real(permute(new_sample{k},[2 1 3 4]))),gather(imag(permute(new_sample{k},[2 1 3 4]))),k,new_sample_id);
            end

end

if frame==1
         for block_id=1:num_feature_blocks
            monument{block_id}=(single(complex(zeros(filter_sz(block_id,1),filter_sz(block_id,2), feature_dim(block_id) ))));
            hf{1,1,block_id}=single(complex(zeros(filter_sz(block_id,1),(filter_sz(block_id,2)+1)/2,params.compressed_dim(block_id))));
            hf1{1,1,block_id}=single(complex(zeros(filter_sz(block_id,1),(filter_sz(block_id,2)+1)/2,params.compressed_dim(block_id))));
            hf2{1,1,block_id}=single(complex(zeros(filter_sz(block_id,1),(filter_sz(block_id,2)+1)/2,params.compressed_dim(block_id))));
         end
         loop_max=10000;
else
         loop_max=10;
end


%% update model %%
train_tracker = (frame < params.skip_after_frame) || (frames_since_last_train >= params.train_gap);
if (train_tracker||frame==1)
                model_update1;     
                 frames_since_last_train = 0;
else
     frames_since_last_train = frames_since_last_train+1;
end
%%%%%%%%%%%%%%%%%%
      
      
      
      
      
      
      
      

                    




