function hf_out = lhs_operation_gpu(hf, mu, lambda, eta, Ap, App,filter_sz, samplesf, reg_filter, sample_weights, feature_reg,binary_mask,patch_mask,frame,zelta,feature_input,cnna,cnnb)
global cnnd;
if length(hf)==5
     feature_input.set_hf_5(real(permute(hf{1},[2 1 3 4])), imag(permute(hf{1},[2 1 3 4])) , real(permute(hf{2},[2 1 3 4])), imag(permute(hf{2},[2 1 3 4])),...
      real(permute(hf{3},[2 1 3 4])), imag(permute(hf{3},[2 1 3 4])) , real(permute(hf{4},[2 1 3 4])), imag(permute(hf{4},[2 1 3 4])) ,real(permute(hf{5},[2 1 3 4])), imag(permute(hf{5},[2 1 3 4]))   ); 
else
     feature_input.set_hf_4(real(permute(hf{1},[2 1 3 4])), imag(permute(hf{1},[2 1 3 4])) , real(permute(hf{2},[2 1 3 4])), imag(permute(hf{2},[2 1 3 4])),...
      real(permute(hf{3},[2 1 3 4])), imag(permute(hf{3},[2 1 3 4])) , real(permute(hf{4},[2 1 3 4])), imag(permute(hf{4},[2 1 3 4])) ); 
end
%%  write gamma and eta into caffe
feature_input.set_parameters(mu,eta); 

cnna.net.forward_prefilled();
% 
output=cnna.net.blobs('conv5_f1');
output_data=output.get_data();
output_data=permute(output_data,[2 1 3 4]);

for block_id=1:length(hf)
      test1=cnna.net.params('conv5_f1',block_id).get_data();
      test1=permute(test1,[2 1 3 4]);
      hf_out{1,1,block_id}=test1(:,:,:,1)+1i*test1(:,:,:,2);
end


end