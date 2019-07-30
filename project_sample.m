function x = project_sample(x, P,x_mean)

if ~isempty(P)
     for k = 1:length(x)
         x{k} = reshape(pagefun(@mtimes, reshape(x{k}, [], size(x{k},3), size(x{k},4)), P{k}), size(x{k},1), size(x{k},2), [], size(x{k},4));
    end
end