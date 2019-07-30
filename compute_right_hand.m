
rhs_samplef = cellfun(@(xf) sum(bsxfun(@times, permute(sample_weights, [2 3 4 1]), permute(xf,[1 2 3 4])), 4), samplesf, 'uniformoutput', false);


rhs_samplef = cellfun(@(xf, yf) bsxfun(@times, (xf), (yf)), rhs_samplef, yf, 'uniformoutput', false); 

rhs_samplef= full_fourier_coeff(rhs_samplef);
 right_hand_tmp=rhs_samplef;

for block_id=1:num_feature_blocks
     tmp=ifftshift(ifftshift(rhs_samplef{block_id},1),2);
     tmp=real(ifft2(tmp));
          tmp=bsxfun(@times,tmp,binary_mask{block_id});
          
          ATXi{block_id}=1*compute_ATXi(Ap{block_id},App{block_id}, tmp,Xi{block_id},filter_sz(block_id,:));
          tmp=tmp-ATXi{block_id};
          tmp=fft2(tmp);
          tmp=fftshift(fftshift(tmp,1),2);
          tmp=tmp(:,1: (filter_sz(block_id,2)+1)/2,: );
          rhs_samplef{1,1,block_id}=gather(tmp);
end