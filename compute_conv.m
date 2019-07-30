function cov=compute_conv(xl,feature_width, feature_height, feature_map_width, feature_map_height,feature_input)


for block_id=1:length(xl)
    
    for i=1:feature_width(block_id)-1
         for j=1:feature_height(block_id)-1
              index=j+(i-1)* (feature_height(block_id)-1);
              if (i==1&&j==1)
                  col_features=zeros(size(xl{block_id},3), (feature_map_height(block_id))*(feature_map_width(block_id)) , ...
                      (feature_width(block_id)-1)*(feature_height(block_id)-1)  );
              end
%               tmp(  i: feature_map_width(block_id)+i, j:feature_map_height(block_id)+j)=tmp(  i: feature_map_width(block_id)+i, j:feature_map_height(block_id)+j)+1;
              tmp=xl{block_id}(  i+1: feature_map_width(block_id)+i, j+1:feature_map_height(block_id)+j,:);
              tmp1=feature_input.im2col(permute(gather(tmp),[2 1 3 4]),1,1,1,1,0,0);
              col_features(:,:,index)=permute(tmp1,[2 1 3  4]);
         end
    end
    col_features=gpuArray(col_features);
    cov{1,1,block_id}=0;
     for i=1:length(index)
         for j=1:length(index)
             cov{1,1,block_id}=cov{1,1,block_id}+col_features(:,:,i)*col_features(:,:,j)';
         end
     end
     a=1;
    
     
     
end