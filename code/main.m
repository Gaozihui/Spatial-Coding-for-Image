
%% Set path and parameters
clear;
close all;
clc;

src_1 = './test images/37967br1.jpg';  
src_2 = './test images/791.jpg';

% src_1 = './test images/4.jpg';  
% src_2 = './test images/Apollo-266.jpg';

% src_1 = './test images/771.jpg';  
% src_2 = './test images/305.jpg';
% 
% src_1 = './test images/Apollo-49.jpg';
% src_2 = './test images/Apollo-266.jpg';


ext = '.sift'; % extension name of SIFT file
siftDim = 128;
maxAxis = 400;


%%  Load image
im_1 = imread(src_1);
if max(size(im_1)) > maxAxis
    im_1 = imresize(im_1, maxAxis / max(size(im_1)));
end

im_2 = imread(src_2);
if max(size(im_2)) > maxAxis
    im_2 = imresize(im_2, maxAxis / max(size(im_2)));
end


%%  Load SIFT feature from file
featPath_1 = [src_1, ext];
featPath_2 = [src_2, ext];

fid_1 = fopen(featPath_1, 'rb');
featNum_1 = fread(fid_1, 1, 'int32');
SiftFeat_1 = zeros(siftDim, featNum_1);
paraFeat_1 = zeros(4, featNum_1);
for i = 1 : featNum_1
    SiftFeat_1(:, i) = fread(fid_1, siftDim, 'uchar');
    paraFeat_1(:, i) = fread(fid_1, 4, 'float32');
end
fclose(fid_1);

fid_2 = fopen(featPath_2, 'rb');
featNum_2 = fread(fid_2, 1, 'int32');
SiftFeat_2 = zeros(siftDim, featNum_2);
paraFeat_2 = zeros(4, featNum_2);
for i = 1 : featNum_2
    SiftFeat_2(:, i) = fread(fid_2, siftDim, 'uchar');
    paraFeat_2(:, i) = fread(fid_2, 4, 'float32');
end
fclose(fid_1);


%%Normalization
SiftFeat_1 = SiftFeat_1 ./ repmat(sqrt(sum(SiftFeat_1.^2)), size(SiftFeat_1, 1), 1);
SiftFeat_2 = SiftFeat_2 ./ repmat(sqrt(sum(SiftFeat_2.^2)), size(SiftFeat_2, 1), 1);


%% Check match based on distances between SIFT descriptors across images
normVal = mean(sqrt(sum(SiftFeat_1.^2)));
matchInd = zeros(featNum_1, 1);
matchDis = zeros(featNum_1, 1);
validDis = [];
gridDisVec = [];
ic = 0;
for i = 1 : featNum_1
    tmpFeat = repmat(SiftFeat_1(:, i), 1, featNum_2);
    d = sqrt(sum((tmpFeat - SiftFeat_2).^2)) / normVal; % L2 distance
    matchDis(i) = min(d);
    [v, ind] = sort(d);
    if v(1) < 0.4            % 最小距离小于0.4，则认为构成一对匹配
        matchInd(i) = ind(1);
        ic = ic + 1;  %构成一对匹配的数量
        validDis(ic, 1 : 3) = [v(1), v(2), v(1) / v(2)]; %保存了每对匹配的V1，V2距离
        tmp = (SiftFeat_1(:, i) - SiftFeat_2(:, ind(1))).^2;
        tmp2 = reshape(tmp(:), 8, 16);
        gridDisVec(ic, 1 : 16) = sqrt(sum(tmp2));
    end
end
figure; stem(matchDis); ylim([0, 1.2]);
figure; stem(matchDis(matchInd > 0)); ylim([0, 1.2]);

%% Put allthe sift into a matrix
matchCount=0;
for i = 1 : featNum_1
    if matchInd(i) > 0
        matchCount = matchCount + 1;
        Q(matchCount,:)=paraFeat_1(:, i)';
        M(matchCount,:)=paraFeat_2(:, matchInd(i))';
    end
end

matchnum=matchCount;
Xq=zeros(matchnum,matchnum);
Yq=zeros(matchnum,matchnum);
Xm=zeros(matchnum,matchnum);
Ym=zeros(matchnum,matchnum);

for i=1:matchnum
    for j=1:matchnum
        if(Q(i,1)<Q(j,1)) 
            Xq(i,j)=0;
        else
            Xq(i,j)=1;
        end
        if (Q(i,2)<Q(j,2))
            Yq(i,j)=0;
        else
            Yq(i,j)=1;
        end
        
        if(M(i,1)<M(j,1)) 
            Xm(i,j)=0;
        else
            Xm(i,j)=1;
        end
        if (M(i,2)<M(j,2))
            Ym(i,j)=0;
        else
            Ym(i,j)=1;
        end
    end
end

Vx=xor(Xq,Xm);

Vy=xor(Yq,Ym);

dismatch=0;

Sx=sum(Vx,2);
Sy=sum(Vy,2);
indx=0;
indy=0;

while 1
    
   if((Sx==0)&(Sy==0))
 
      break;%跳出循环
 
   end
    
    Sx=sum(Vx,2);
    [mx, indx] = max(Sx);
    if mx>=1
    dismatch=dismatch+1;
    IND(dismatch)=indx;
    end
    Vx(:,indx)=0;
    Vx(indx,:)=0;
    Vy(:,indx)=0;
    Vy(indx,:)=0;

    
    Sy=sum(Vy,2);
    [my, indy] = max(Sy);
    if my>=1
    dismatch=dismatch+1;
    IND(dismatch)=indy;
    end
    Vx(:,indy)=0;
    Vx(indy,:)=0;
    Vy(:,indy)=0;
    Vy(indy,:)=0;
    


end

[row0,col0] = find(matchInd>0);

for i=1:length(IND)
    del(i)=row0(IND(i));
end


%% Show the local matching results on RGB image
[row, col, cn] = size(im_1);
[r2, c2, n2] = size(im_2);
imgBig = 255 * ones(max(row, r2), col + c2, 3);
imgBig(1 : row, 1 : col, :) = im_1;
imgBig(1 : r2, col + 1 : end, :) = im_2;
np = 40;
thr = linspace(0,2*pi,np) ;
Xp = cos(thr);
Yp = sin(thr);
paraFeat_2(1, :) = paraFeat_2(1, :) + col;
figure(3); imshow(uint8(imgBig)); axis on;
hold on;
matchCount = 0;
num_r=0;
num_b=0;


for i = 1 :1: featNum_1

    if matchInd(i) > 0
        matchCount = matchCount + 1;
        xys = paraFeat_1(:, i); %xy是读取的特征的位置信息
        xys2 = paraFeat_2(:, matchInd(i));
        figure(3);
        hold on;
     
      if(find(del==i))
        plot([xys(1), xys2(1)], [xys(2), xys2(2)], '-r', 'LineWidth', 0.8);
        plot(xys(1) + Xp * xys(3) * 6, xys(2) + Yp * xys(3) * 6, 'r');%画圆标出特征
        plot(xys2(1) + Xp * xys2(3) * 6, xys2(2) + Yp * xys2(3) * 6, 'r');
        num_r=num_r+1;
      else
     
     num_b=num_b+1;
       plot([xys(1), xys2(1)], [xys(2), xys2(2)], '-b', 'LineWidth', 0.8);
        plot(xys(1) + Xp * xys(3) * 6, xys(2) + Yp * xys(3) * 6, 'b');%画圆标出特征
        plot(xys2(1) + Xp * xys2(3) * 6, xys2(2) + Yp * xys2(3) * 6, 'b');
      end
      
     
    end
       
%         figure(4);
%         clf;
%         subplot(311); stem(SiftFeat_1(:, i)); xlim([0, 128]); ylim([0, 0.5]);
%         title(sprintf('Feature pair %d', matchCount));
%         subplot(312); stem(SiftFeat_2(:, matchInd(i))); xlim([0, 128]); ylim([0, 0.5]);
%         tmp = SiftFeat_1(:, i) - SiftFeat_2(:, matchInd(i));
%         subplot(313); stem(tmp); xlim([0, 128]);
%         title('Difference between the above two features per dimension');
%         disVal = sum(sqrt(tmp.^2));
%         ylim([0, 0.2]);
    
end
title(sprintf('Total local matches : %d (%d-%d)', length(find(matchInd)), featNum_1 ,featNum_2));
%  title(sprintf('error matches : %d ',num_r ));
% title(sprintf('correct matches : %d ',num_b ));

% figure(3);
% title(sprintf('Total local matches : %d (%d-%d)', length(find(matchInd)), featNum_1 ,featNum_2));
% hold off;

