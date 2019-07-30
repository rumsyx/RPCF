close all
clear
clc
warning off all;
addpath 'libsvm'
%%�������п�
addpath('caffe-fcnt/matlab/caffe/', 'util');
addpath(genpath('benchmark/'));
 seqs=configSeqs;

trackers=configTrackers;

shiftTypeSet = {'left','right','up','down','topLeft','topRight','bottomLeft','bottomRight','scale_8','scale_9','scale_11','scale_12'};

evalType='OPE'; %'OPE','SRE','TRE'

% diary(['./tmp/' evalType '.txt']);

numSeq=length(seqs);
numTrk=length(trackers);

finalPath = ['benchmark/results/' evalType '/'];

if ~exist(finalPath,'dir')
    mkdir(finalPath);
end

tmpRes_path = ['benchmark/tmp/' evalType '/'];
bSaveImage=0;

if ~exist(tmpRes_path,'dir')
    mkdir(tmpRes_path);
end

pathAnno = 'benchmark/anno/';

for idxSeq=1:length(seqs)
%     try
    s = seqs{idxSeq};
    
%      if ~strcmp(s.name, 'coke')
%         continue;
%      end
        
    s.len = s.endFrame - s.startFrame + 1;
    s.s_frames = cell(s.len,1);
    nz	= strcat('%0',num2str(s.nz),'d'); %number of zeros in the name of image
    for i=1:s.len
        image_no = s.startFrame + (i-1);
        id = sprintf(nz,image_no);
        s.s_frames{i} = strcat(s.path,id,'.',s.ext);
    end
    
    img = imread(s.s_frames{1}); 
    [imgH,imgW,ch]=size(img);
    
    rect_anno = dlmread([pathAnno s.name '.txt']);
    numSeg = 20;
    
    [subSeqs, subAnno]=splitSeqTRE(s,numSeg,rect_anno);
    
    switch evalType
        case 'SRE'
            subS = subSeqs{1};
            subA = subAnno{1};
            subSeqs=[];
            subAnno=[];
            r=subS.init_rect;
            
            for i=1:length(shiftTypeSet)
                subSeqs{i} = subS;
                shiftType = shiftTypeSet{i};
                subSeqs{i}.init_rect=shiftInitBB(subS.init_rect,shiftType,imgH,imgW);
                subSeqs{i}.shiftType = shiftType;
                
                subAnno{i} = subA;
            end

        case 'OPE'
            subS = subSeqs{1};
            subSeqs=[];
            subSeqs{1} = subS;
            
            subA = subAnno{1};
            subAnno=[];
            subAnno{1} = subA;
        otherwise
    end

            
    for idxTrk=1:1
       t.name='RPCF';
       t.namePaper='RPCF';

%         if ~strcmp(t.name, 'LSK')
%             continue;
%         end

        % validate the results
      %% �����Լ��Ľ���ļ���
        if exist([finalPath s.name '_' 'RPCF' '.mat'])
            load([finalPath s.name '_' 'RPCF' '.mat']);
            bfail=checkResult(results, subAnno);
            if bfail
                disp([s.name ' '  t.name]);
            end
            continue;
        end

        switch t.name
            case {'VTD','VTS'}
                continue;
        end

        results = [];
%         load(['C:\st_max_margin_4\benchmark\results_backup\TRE\faceocc2_my.mat']);
        for idx=1:length(subSeqs)
%             for idx=7:10
%             if (subAnno{idx}(1,3)<70||subAnno{idx}(1,4)<70)&&(~(imgW>600&&imgH>400))
                
%       for idx=8:10
            disp([num2str(idxTrk) '_' t.name ', ' num2str(idxSeq) '_' s.name ': ' num2str(idx) '/' num2str(length(subSeqs))])       

            rp = [tmpRes_path s.name '_' t.name '_' num2str(idx) '/'];  %�洢���·��
            if bSaveImage&~exist(rp,'dir')
                mkdir(rp);
            end
            
            subS = subSeqs{idx};
            
            subS.name = [subS.name '_' num2str(idx)];
            
%             subS.s_frames = subS.s_frames(1:20);
%             subS.len=20;
%             subS.endFrame=subS.startFrame+subS.len-1;
            
%                 funcName = ['res=run_' t.name '(subS, rp, bSaveImage);'];    
%                 eval(funcName);
              t1=clock;
              res_path=0; bSaveImage=0;
              [ mm] = run_my(subS,  res_path, bSaveImage );
%              [mm, fps] = run_my(subS.name(1:end-2),subS);
              t2=etime(clock,t1);
%               res.fps=fps;
              res.res=mm;
              res.type='rect';
                  res.len = subS.len;
            res.annoBegin = subS.annoBegin;
            res.startFrame = subS.startFrame;
            results{idx} = res;
%               else
%                  continue;
%               end
         end
                
              
           
            
        
                    
%             switch evalType
%                 case 'SRE'
%                     res.shiftType = shiftTypeSet{idx};
%             end
            
            
            
        end
        save([finalPath s.name '_' t.name '.mat'], 'results');
%         catch
%           a=1;
%        end
    end


figure
t=clock;
t=uint8(t(2:end));
disp([num2str(t(1)) '/' num2str(t(2)) ' ' num2str(t(3)) ':' num2str(t(4)) ':' num2str(t(5))]);

