function [ res] = run_my(seq,  res_path, bSaveImage )
close all

seq.name = seq.path(25:end-5);
init=seq.init_rect;
seq.init_rect = init(1,:);
seq.ground=init;
res = RPCF(seq);
% res=res.res;
caffe.reset_all();
end

