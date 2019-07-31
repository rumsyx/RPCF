%% demo_RPCF

clear all;
clc;
% seq_name = 'skiing';
% seq_name = 'car4';
% seq_name = 'Diving';
% seq_name = 'DragonBaby';
% seq_name = 'KiteSurf';
% seq_name = 'girl';
% seq_name = 'Board';
% seq_name = 'freeman4';
% seq_name = 'DragonBaby';
% seq_name = 'fleetface';
%  seq_name = 'Jump';
% seq_name = 'Twinnings';
% seq_name = 'bolt';
% seq_name = 'skiing';
% seq_name = 'Bolt2'; 
% seq_name = 'soccer'; 
% seq_name = 'basketball';
% seq_name = 'skating1';
% seq_name = 'carScale';
% seq_name = 'couple';
% seq_name = 'jogging';
% seq_name = 'Coupon';
% seq_name = 'Girl2';
seq_name = 'freeman4';
%  seq_name = 'matrix';
%  seq_name = 'ironman';
%  seq_name = 'shaking';
% seq_name = 'singer1';
% seq_name = 'lemming';
% seq_name = 'Bird1';
% seq_name = 'bird2';
% seq_name = 'Vase';
% seq_name = 'tiger2_c';
%  seq_name = 'singer2'; 
% seq_name = 'motorRolling'; 
% seq_name='liquor';
% seq_name = 'ClifBar';
% seq_name = 'freeman1';
% seq_name = 'suv';
% seq_name = 'Box';
%% download video data if necessary
% if ~exist(['./video/' seq_name '/img/0001.jpg'], 'file')
%     system('sh ./video/download_basketball.sh');
% end
%% 
% seq_info = load(['video/' seq_name '/info.txt']);

mainpath = '/home/rum/OTB-100-PCF/';

seq.name = seq_name;
seq.path = char(strcat(mainpath ,seq.name,'/img/'));
init=load([mainpath,seq_name,'/groundtruth_rect.txt']);
seq.endFrame = length(init); seq.startFrame = 1; seq.nz = 4; seq.ext ='jpg';seq.init_rect = [0,0,0,0];
seq.len = seq.endFrame - seq.startFrame + 1;
seq.s_frames = cell(seq.len,1);
nz	= strcat('%0',num2str(seq.nz),'d'); %number of zeros in the name of image
for i=1:seq.len
    image_no = seq.startFrame + (i-1);
    id = sprintf(nz,image_no);
    seq.s_frames{i} = strcat(seq.path,id,'.',seq.ext);
end
seq.init_rect = init(1,:);
seq.ground=seq.init_rect;
res = RPCF(seq);
% pause(10);