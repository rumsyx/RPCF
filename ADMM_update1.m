global H_total;

compute_right_hand1;

if frame==1
   rho1=[]; p1=[];
end

if frame>1&&ite_out==2
    max_CG_iter=2;
end
feature_input.set_resolution_index(2);
feature_input.set_factor(0.5);
% max_CG_iter=100;
     [hf1, flag, relres, iter, res_norms, p1, rho1] = pcg_ccot(...
        @(x) lhs_operation_gpu(x, gamma1, lambda1, eta1, Ap1, App1, filter_sz, samplesf_tmp, reg_filter, sample_weights, feature_reg,binary_mask,patch_mask,frame,Xi1,feature_input,cnna,cnnb),...
        rhs_samplef, CG_tol, max_CG_iter, ...
        [], ...
         [], hf1, p1, rho1,cnn_third,feature_input);
     
 if length(hf)==4
   [H_masked{1},H_masked{2},H_masked{3},H_masked{4}]=feature_input.get_H_masked_4();
   H_masked{1}=permute(H_masked{1},[2 1 3 4]);
   H_masked{2}=permute(H_masked{2},[2 1 3 4]);
   H_masked{3}=permute(H_masked{3},[2 1 3 4]);
   H_masked{4}=permute(H_masked{4},[2 1 3 4]);
else
    [H_masked{1},H_masked{2},H_masked{3},H_masked{4},H_masked{5}]=feature_input.get_H_masked_5();
    H_masked{1}=permute(H_masked{1},[2 1 3 4]);
    H_masked{2}=permute(H_masked{2},[2 1 3 4]);
    H_masked{3}=permute(H_masked{3},[2 1 3 4]);
    H_masked{4}=permute(H_masked{4},[2 1 3 4]);
    H_masked{5}=permute(H_masked{5},[2 1 3 4]);
end  
     
     %% update Xi
     for block_id=1:num_feature_blocks
         if block_id~=3
        AW_MC1{block_id}=compute_AW_MC(Ap1{block_id},H_masked{block_id},filter_sz(block_id,:));
         else
             AW_MC1{block_id}=compute_AW_MC(Ap{block_id},H_masked{block_id},filter_sz(block_id,:));
         end
%         if ite_out==1
         if block_id~=3
          Xi1{block_id}=Xi1{block_id}+0.5*gamma1*(AW_MC1{block_id});
         else
             Xi1{block_id}=Xi1{block_id}+0.5*gamma1*(AW_MC1{block_id});
         end
%         end
     end
     gamma1 = min(betha1 * gamma1, gammamax1);
     
     