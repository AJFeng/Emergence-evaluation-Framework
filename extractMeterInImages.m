path=load('segment_file_address.mat');
myFolder = load('Img_folder_address.mat');

path=path.path;
myFolder =myFolder.myFolder;
%myFolder = ('E:\matlab\0531phantom\Phantom_4\data3\');
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

filePattern = fullfile(myFolder);
jpegFiles = dir(filePattern);
gps=zeros(length(jpegFiles)-3,2);

for k = 3:length(jpegFiles) %length(jpegFiles)
    imgInfo=imfinfo(strcat(jpegFiles(k).folder,'\',jpegFiles(k).name));
  
    long=imgInfo.GPSInfo.GPSLongitude;
    long_tranform=-(long(1,1)+long(1,2)/60+long(1,3)/3600);
    lat=imgInfo.GPSInfo.GPSLatitude;
    lat_tranform=lat(1,1)+lat(1,2)/60+lat(1,3)/3600;
    gps(k-2,:)=[lat_tranform long_tranform];
end


gps_split=[];
gps_split_name=[];

lat=0.000009;  %36.41236607
long=0.000011;  %-89.69620545

%t = cputime;
tic


row_space=[];
for kk=3:size(jpegFiles,1)-1
    img_gps=gps(kk-2,:);
    p1=imread(strcat(jpegFiles(kk).folder,'\',jpegFiles(kk).name));
    p3=p1;

    for i=1:size(p3,1)    % removed the effect of PVC
        for j=1:size(p3,2)
            if (p3(i,j,1)>200)||(p3(i,j,3)>200)
              p3(i,j,:)=0;
            end
        end
    end
    % imtool(p3);

    p3p=decorrstretch(p3);
    p4p=p3p(:,:,2);
    p4p(p4p<180)=0;
    % imtool(p4p);


    L=p4p;
    %L=bwareaopen(L, 50);
    % hough line detection
    [H,T,R] = hough(L,'RhoResolution',1,'Theta',-90:0.5:89);

    P  = houghpeaks(H,15,'threshold',ceil(0.01*max(H(:))));
    x = T(P(:,2)); y = R(P(:,1));
%     plot(x,y,'s','color','white');

    lines = houghlines(L,T,R,P,'FillGap',8,'MinLength',7);
%     figure, imshow(L), hold on
%     for k = 1:length(lines)
%        xy = [lines(k).point1; lines(k).point2];
%        plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% 
%        % Plot beginnings and ends of lines
%        plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%        plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% 
%     end
%     
    
    M=mode(x); %the cotton row angle
    
    img_rotate = imrotate(p1,M);
    L = imrotate(L,M);
    
    % hough line detection
    [H,T,R] = hough(L,'RhoResolution',1,'Theta',-7:7:7);

    P  = houghpeaks(H,15,'threshold',ceil(0.01*max(H(:))));
    x = T(P(:,2)); y = R(P(:,1));
%     plot(x,y,'s','color','white');
    for tt=size(x,2):-1:1
        if x(tt)~=0
            P(tt,:)=[];
        end
    end

    x = T(P(:,2)); y = R(P(:,1));
    lines = houghlines(L,T,R,P,'FillGap',8,'MinLength',7);
%     gcf=figure('visible','off'), imshow(L), hold on
%     figure, imshow(L),impixelinfo; hold on
%     for k = 1:length(lines)
%        xy = [lines(k).point1; lines(k).point2];
%        plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% 
%        % Plot beginnings and ends of lines
%        plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%        plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% 
%     end
    
    y2=sort(y');
    y3=y2(2:end);
%     this_row=padcat(1:15,(y3-y2(1:end-1))');
%     row_space=[row_space;this_row(2,:)];
    
%     saveas(gcf,strcat(path,jpegFiles(kk).name,'.png'));
    
    y3=sort(y3-y2(1:end-1));
    
    while (y3(end-1)>800)&(y3(1)<800)
        y3(end)=[];
    end
    row_space=y3(end-1); % row_space pixel/meter
    Cx=size(img_rotate,1)/2;
    Cy=size(img_rotate,2)/2;
    
    %figure, imshow(img_rotate), hold on
    %plot(Cy,Cx,'x','LineWidth',20,'Color','blue');
    
    gps_one=[0.0,0.0];
    for split_id=-2:2
        for row_id=3:(size(y2)-2)
            gps_one(2)=img_gps(2)+(y2(row_id)-Cy)/row_space*long;
            gps_one(1)=img_gps(1)-split_id*lat;
            fName=strcat(jpegFiles(kk).name,'_',num2str(split_id),'_',num2str(row_id),'_',num2str(img_gps(1)),'_',num2str(img_gps(2)),'.tiff');
            
            top=Cx+split_id*row_space;
            
            if (top+row_space)>size(img_rotate,1)
                top=size(img_rotate,1)-row_space-1;
            elseif (top+row_space)<1
                top=1;
            end
                    
            if row_id<size(y2,1)/2
                left=max(y2(row_id)-row_space/2,1);
                temp=img_rotate(top:top+row_space,left:left+row_space,:);
            else
                right=min(y2(row_id)+row_space/2,size(img_rotate,2));
                temp=img_rotate(top:top+row_space,right-row_space:right,:);
            end
            imwrite(temp,strcat(path,fName),'tiff','Compression', 'none');
            gps_split=[gps_split;gps_one];
            gps_split_name=[gps_split_name; string(fName)];
        end
    end
    
end
toc
%save 'gps_split.mat' gps_split;
%save 'gps_split_name.mat' gps_split_name;
%e = cputime-t;  % seconds

result=[gps_split,gps_split_name];
T = array2table(result,'VariableNames',{'lat','long','name'});
writetable(T,'gps.csv');