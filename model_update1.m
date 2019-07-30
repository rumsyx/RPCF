new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf, 'uniformoutput', false); 
reg_energy{1,1,1}=0.1; reg_energy{1,1,2}=0.1; reg_energy{1,1,3}=0.1; reg_energy{1,1,4}=0.1; %reg_energy{1,1,5}=0.1;


feature_input.set_frame_id(frame);
if frame == 1 
        %% Initialize Conjugate Gradient parameters
        p = [];
        rho = [];
        p1 = [];
        rho1 = [];
        p2 = [];
        rho2 = []; 
        
        max_CG_iter = params.init_max_CG_iter;
        sample_energy = new_sample_energy;
 else
        max_CG_iter = params.max_CG_iter;
        
        if params.CG_forgetting_rate == inf || params.learning_rate >= 1
          
            max_CG_iter=params.init_max_CG_iter;
        else
            rho = rho / (1-params.learning_rate)^params.CG_forgetting_rate;
            rho1 = rho1 / (1-params.learning_rate)^params.CG_forgetting_rate;
            rho2 = rho2 / (1-params.learning_rate)^params.CG_forgetting_rate;
%             rho1=(1-params.learning_rate)^20/ ((1-params.learning_rate)^params.CG_forgetting_rate);
%             p{3}=p{3}*rho1;
        end
        
        % Update the approximate average sample energy using the learning
        % rate. This is only used to construct the preconditioner.
        sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, sample_energy, new_sample_energy, 'uniformoutput', false);
 end
% diag_M = cellfun(@(m, reg_energy) (1-params.precond_reg_param) * bsxfun(@plus, params.precond_data_param * m, (1-params.precond_data_param) * mean(m,3)) + params.precond_reg_param*reg_energy, sample_energy, reg_energy, 'uniformoutput',false);

samplesf_tmp=samplesf;

if frame==1
    out_loop=2;
%     inner_loop=5;
    lb = 0.5*ones(9,1);
    ub =1.5*ones(9,1);
    reliability_val=ones(params.frag_num,1);
      
else
    out_loop=2;
%     inner_loop=5;
end

%% inialize the parameters for the model
gamma=0.55;
gamma1=0.6;
gamma2=0.10;
lambda=1;
lambda1=1;
lambda2=1;
eta=0.01;
eta1=0.01;
eta2=0.01;
betha = 10;
betha1 = 10;
betha2 = 10;
gammamax = 10000;
gammamax1 = 10000;
gammamax2 = 10000;
if frame==1
   for block_id=1:num_feature_blocks
              %%  construct index for computing matrix A (in paper the matrix is denoted as V).
        [Ap{block_id}, App{block_id}]=inialized_matrix_A(filter_sz(block_id,:),binary_mask{block_id},2);
        [Ap1{block_id}, App1{block_id}]=inialized_matrix_A(filter_sz(block_id,:),binary_mask{block_id},3);
       [ATAW_positive_index{block_id}, ATAW_negative_index{block_id}]=construct_index_ATAW(Ap{block_id});
        [ATAW_positive_index1{block_id}, ATAW_negative_index1{block_id}]=construct_index_ATAW(Ap1{block_id});    
           
           
        feature_input.input_App(permute(Ap{block_id}-1,[2 1 3 4]), permute(Ap1{block_id}-1,[2 1 3 4]),block_id-1,...
            permute(App{block_id},[2 1 3 4]),permute(App1{block_id},[2 1 3 4] ),  ...
            permute(ATAW_positive_index{block_id},[2 1 3 4])-1 ,permute(ATAW_negative_index{block_id},[2 1 3 4])-1, ...
            permute(ATAW_positive_index1{block_id},[2 1 3 4])-1 ,permute(ATAW_negative_index1{block_id},[2 1 3 4])-1);
%         [Ap{block_id}, App{block_id}]=inialized_matrix_A(filter_sz(block_id,:),binary_mask{block_id},4);
      %% inialized Xi
        Xi{block_id}=zeros(size(Ap{block_id},1)*feature_dim(block_id),1);
        Xi1{block_id}=zeros(size(Ap1{block_id},1)*feature_dim(block_id),1);
        
        if block_id==3
            Xi1{block_id}=zeros(size(Ap{block_id},1)*feature_dim(block_id),1);
        end
        
   end
end


for block_id=1:num_feature_blocks
   Xi{block_id}=Xi{block_id}*0;
   Xi1{block_id}=Xi1{block_id}*0;
end
if frame==37
    a=1;
end

% tic;
for ite_out=1:out_loop   
      ADMM_update;         %% pooling kernel=2*2  THE ONLY KERNEL SIZE ADOPTED BY RPCF.  
%     ADMM_update1;      %% pooling kernel=3*3
%     ADMM_update2;      %% pooling kernel=1*1

end


hff= symmetrize_filter(hf);
hff=full_fourier_coeff(hff);
for block_id=1:num_feature_blocks

             hf_tmp=ifftshift(ifftshift(hff{block_id},1),2);
              H=real(ifft2(hf_tmp));
             if (block_id==3)
                 H=H*1.2;
             end
             H=bsxfun(@times,H,binary_mask{block_id}(:,:,1));
             hff{block_id}=fft2(H);
             hff{block_id}=fftshift(fftshift(hff{block_id},1),2);
end
clear H;
weight_f= hff;

   