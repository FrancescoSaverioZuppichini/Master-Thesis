clc
clear all
close all

path = '/home/francesco/Documents/krock-sim/krock/krock2_ros/map_generation/quarry.wbt';  


% spacing between nodes (pixel size)
spacing=0.02;

% random terrain
Terrain=rand(500,500)*0.3;
Terrain=imgaussfilt(Terrain, 2);

% Read txt into a cell
fid = fopen(path,'r');
i = 1; 
correct_terrain=0;
terrain_done=0;
tline = fgetl(fid);
A{i} = tline;
while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    A{i} = tline;
    % get the start of the height matrix
    if strcmp(tline,'      geometry DEF EL_GRID ElevationGrid {')
        l1=i+1;
    end
    % end of the matrix
    if strcmp(tline,'  name "terrain"')
        l2=i;
        terrain_done=1;
    end
end
fclose(fid);


% Insert new terrain height
for i=1:size(Terrain,1)
    B{i}=num2str(Terrain(i,:));
end
A{l2-8}=[']'];
A{l2-7}=['        xDimension ' num2str(size(Terrain, 2))];
A{l2-6}=['        xSpacing ' num2str(spacing)];
A{l2-5}=['        zDimension ' num2str(size(Terrain, 1))];
A{l2-4}=['        zSpacing ' num2str(spacing)];
A=[A(1:l1) B A(l2-8:end)];



% Write modified world file
fid = fopen(path, 'w');
for i = 1:numel(A)
    if A{i+1} == -1
        fprintf(fid,'%s', A{i});
        break
    else
        fprintf(fid,'%s\n', A{i});
    end
end

