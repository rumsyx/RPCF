function [roi, roi_pos, preim, pad,roi1] = ext_roi_backup(im, GT, l_off, roi_size, r_w_scale)
[h, w, ~] = size(im);
win_w = GT(3);
win_h = GT(4);
win_lt_x = GT(1);
win_lt_y = GT(2);
win_cx = round(win_lt_x+win_w/2+l_off(1));
win_cy = round(win_lt_y+win_h/2+l_off(2));
roi_w = r_w_scale(1)*win_w;
roi_h = r_w_scale(2)*win_h;
x1 = win_cx-round(roi_w/2);
y1 = win_cy-round(roi_h/2);
x2 = win_cx+round(roi_w/2);
y2 = win_cy+round(roi_h/2);

im = double(im);
clip = min([x1,y1,h-y2, w-x2]);

%% 找到目标的位置，然后将目标反置
im1=im;
GT=floor(GT);

pad = 0;
if clip<=0
    pad = abs(clip)+1;
    im = padarray(im, [pad, pad]);
    im1= padarray(im1, [pad, pad]);
    x1 = x1+pad;
    x2 = x2+pad;
    y1 = y1+pad;
    y2 = y2+pad;
end
roi =  imresize(im(y1:y2, x1:x2, :), [roi_size, roi_size]);
preim = zeros(size(im,1), size(im,2));
roi_pos = [x1, y1, x2-x1+1, y2-y1+1];

% marginl = floor((roi_warp_size-roi_size)/2);
% marginr = roi_warp_size-roi_size-marginl;

% roi = roi(marginl+1:end-marginr, marginl+1:end-marginr, :);
% roi = imresize(roi, [roi_size, roi_size]);
end