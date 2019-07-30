function hf_out = lhs_operation(hf, mu, lambda, eta, Ap, App,filter_sz, samplesf, reg_filter, sample_weights, feature_reg,binary_mask,patch_mask,frame,zelta,feature_input,cnna)


global H_total;
num_features = length(hf);
output_sz = [size(hf{1},1), 2*size(hf{1},2)-1];
pad_sz = cellfun(@(hf) (output_sz - [size(hf,1), 2*size(hf,2)-1]) / 2, hf, 'uniformoutput',false);



hf_full=full_fourier_coeff(hf);
for block_id=1:length(hf_full)
   tmp=ifftshift(ifftshift(hf_full{block_id},1),2);
   tmp1=real(ifft2(tmp));
   tmp2=bsxfun(@times,tmp1,binary_mask{block_id});
   H_total{block_id}=real(tmp2);
 %% we obtain H for each feature layer
   ATAW_MC{block_id}=compute_ATAW_MC(Ap{block_id},App{block_id},tmp2,filter_sz(block_id,:));
   %% than we compute AT{zelta}
%    ATZ{block_id}=compute_ATZ(Ap{block_id},tmp2,zelta{block_id},filter_sz(block_id,:));
   
   tmp3=fft2(tmp2);
   tmp4=fftshift(fftshift(tmp3,1),2);
   hf_tmp{1,1,block_id}=tmp4(:,1:(size(tmp2,2)+1)/2,:); 
end  

sh_cell= cellfun(@(hf,samplesf)   permute(  sum(bsxfun(@times, permute(conj(samplesf),[3 4 2 1]), hf),3), [4 3 1 2]   )  , hf_tmp, samplesf, 'uniformoutput', false);

% sum over all feature blocks
sh = sh_cell{1};    % assumes the feature with the highest resolution is first
for k = 2:num_features
    sh(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
        sh(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) + sh_cell{k};
end

% weight all the samples
sh = bsxfun(@times,sample_weights,sh);

sh=permute(sh,[3 4 2 1]);

hf_out = cellfun(@(samplesf,pad_sz)   sum(bsxfun(@times,   repmat( sh(1+pad_sz(1):end-pad_sz(1), 1+pad_sz(2):end,1,:) ,[1,1,size(samplesf,2),1] ) ,permute(samplesf,[3 4 2 1] )   ),4)  , ...
    samplesf, pad_sz, 'uniformoutput', false);

hf_out_full= symmetrize_filter(hf_out);
hf_out_full=full_fourier_coeff(hf_out_full);
% tic;
for block_id=1:num_features
    tmp=ifftshift(ifftshift(hf_out_full{block_id},1),2);
    tmp=real(ifft2(tmp));
    tmp=bsxfun(@times,tmp,binary_mask{block_id});
    %% we sum up the filter weights and the lagarange multiplier
    tmp=tmp+eta*H_total{block_id}+mu*ATAW_MC{block_id};
    
    tmp=fft2(tmp);
    hf_out_full{1,1,block_id}=fftshift(fftshift(tmp,1),2);
end

hf_out = cellfun(@(hf) hf(:,1:(size(hf,2)+1)/2,:), hf_out_full, 'uniformoutput', false);

end