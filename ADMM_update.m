global H_total;
feature_input.set_resolution_index(1);
compute_right_hand;

if frame==1
   rho=[]; p=[];
end

if frame>1&&ite_out==2
    max_CG_iter=2;
end

% max_CG_iter=100;

feature_input.set_factor(0.3);

     [hf, flag, relres, iter, res_norms, p, rho] = pcg_ccot(...
        @(x) lhs_operation_gpu(x, gamma, lambda, eta, Ap, App, filter_sz, samplesf_tmp, reg_filter, sample_weights, feature_reg,binary_mask,patch_mask,frame,Xi,feature_input,cnna,cnnb),...
        rhs_samplef, CG_tol, max_CG_iter, ...
        [], ...
         [], hf, p, rho,cnn_third,feature_input);
     
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
        AW_MC{block_id}=compute_AW_MC(Ap{block_id},H_masked{block_id},filter_sz(block_id,:));
%         if ite_out==1
       if block_id~=3
          Xi{block_id}=Xi{block_id}+0.5*gamma*(AW_MC{block_id});
       else
           Xi{block_id}=Xi{block_id}+0.5*gamma*(AW_MC{block_id});
       end
%         end
     end
     gamma = min(betha * gamma, gammamax);
     
     