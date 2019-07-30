xl1 = cellfun(@(x) reshape(x, [], size(x,3)), xl, 'uniformoutput', false);
x_mean=cellfun(@(x) permute(mean(x, 1),[1 3 2]), xl1, 'uniformoutput', false);

for block_id=1:length(xl)
    tmp=repmat(x_mean{block_id},[feature_sz(block_id,1)  feature_sz(block_id,2) 1]);
    x_mean{1,1,block_id}=tmp;
end

%%  Get the target size under each block_id
for block_id=1:num_feature_blocks
     [yy,xx]=find(binary_mask{block_id});
     x_min=min(xx(:)); y_min=min(yy); x_max=max(xx(:)); y_max=max(yy(:));
     feature_width(block_id)=x_max-x_min+1; feature_height(block_id)=y_max-y_min+1;
     feature_map_width(block_id)=feature_sz(block_id,1)-feature_width(block_id)+1; 
     feature_map_height(block_id)=feature_sz(block_id,1)-feature_height(block_id)+1; 
end

cov=compute_conv(xl,feature_width, feature_height, feature_map_width, feature_map_height,feature_input);

 %% Constructed covariance matrix according to num_map and xl1 value.

[projection_matrix, ~, ~] = cellfun(@(x) svd(x), cov, 'uniformoutput', false);
% [projection_matrix, ~, ~] = cellfun(@(x) svd(x' * x), xl1, 'uniformoutput', false);
for block_id=1:length(xl)
     compressed_dim_cell{1,1,block_id}=params.compressed_dim(block_id);
end
projection_matrix = cellfun(@(P, dim) single(P(:,1:dim)/5), projection_matrix, compressed_dim_cell, 'uniformoutput', false);