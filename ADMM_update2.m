global H_total;

compute_right_hand2;

if frame==1
%    rho2=[]; p2=[];
end

if frame>1&&ite_out==2
    max_CG_iter=2;
end

% max_CG_iter=100;
     [hf2, flag, relres, iter, res_norms, p2, rho2] = pcg_ccot(...
        @(x) lhs_operation_gpu1(x, lambda2, eta2, filter_sz, samplesf_tmp, reg_filter, sample_weights, feature_reg,binary_mask,patch_mask,frame,feature_input,cnna,cnnb),...
        rhs_samplef, CG_tol, max_CG_iter, ...
        [], ...
         [], hf2, p2, rho2,cnn_third,feature_input);

     
     