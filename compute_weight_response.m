function [response_1,right_hand_r1]=compute_weight_response(hf,samplesf_tmp,sample_weights,y_real,num_feature_blocks,pad_sz)

sh_cell = cellfun(@(hf,samplesf) sum(bsxfun(@times, samplesf,conj(hf)),3), hf, samplesf_tmp, 'uniformoutput', false);
sh = sh_cell{1};    % assumes the feature with the highest resolution is first
for k = 2:num_feature_blocks
           sh(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end,1,:) = ...
                  sh(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end,1,:) + sh_cell{k};
end
sh1 = bsxfun(@times,permute(sample_weights,[2,3,4,1]),sh);
sh = bsxfun(@times,permute(sqrt(sample_weights),[2 3 4 1]),sh);
sh= symmetrize_filter1(sh);
sh_full=full_fourier_coeff(permute(sh,[1,2,4,3]));
sh1= symmetrize_filter1(sh1);
sh_full1=full_fourier_coeff(permute(sh1,[1,2,4,3]));
response_1=real(ifft2(  ifftshift(ifftshift(sh_full,1),2)  ));
tmp2=real(ifft2(  ifftshift(ifftshift(sh_full1,1),2)  ));
right_hand_r1=bsxfun(@times,tmp2,y_real{1});
right_hand_r1=sum(right_hand_r1(:));
