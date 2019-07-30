    for block_id=1:num_feature_blocks
                tmp1{block_id}=sum(bsxfun(@times,conj(samplesf{block_id}),(weight_f{block_id})),3);
    end
   aa=tmp1{1}+tmp1{2}+padarray(tmp1{3},pad_sz{3})+tmp1{4}+tmp1{5}-yf;
   weight_diff=cnna.net.params('conv5_f1',3).get_diff();
   weight_diff=permute(weight_diff,[2 1 3 4]);
   aa1=aa(:,:,1,1);
   aa1=aa1(pad_sz{3}(1)+1:end-pad_sz{3},pad_sz{3}(1)+1:end-pad_sz{3});
   aa1=repmat(aa1,[1 1 512]);
   aa2=samplesf{3}(:,:,:,1);
   aa2=aa2.*aa1;
   a=1;