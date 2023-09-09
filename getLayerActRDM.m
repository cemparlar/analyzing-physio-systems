function [RDM_mat] = getLayerActRDM(unique_obj_names,obj_names,arrayLayerActivation,unique_cat_names,cat_numbers,obj_numbers,titleplot)

average_per_obj = [];
for i = 1:length(unique_obj_names) 
    object_idx = find(obj_names == unique_obj_names(i));
    object = arrayLayerActivation(object_idx,:);
    %all the neural responses for each object     
    average_per_obj(i,:) = mean(object);
    %mean of the neural responses; organized as in unique_obj 
end

% Right order of objects
unique_obj_organized = [];
row = 1;
for i = 1:length(unique_cat_names)
    all_idx_of_cat = find(cat_numbers == i); %%all indices of category i
    obj_disorganized = obj_numbers(all_idx_of_cat);
    obj_organized = unique(sort(obj_disorganized));
    unique_obj_organized(row:row+7,1) = obj_organized;
    row = row+8;
end

% Right order of average neural responses to objects
organized_responses = [];
for i = 1:length(unique_obj_organized)
    idx_IT_responses = unique_obj_organized(i);
    organized_responses(i,:) = average_per_obj(idx_IT_responses,:);   
end

mat = organized_responses';

%get correlation matrix

one_mat = ones(64, 64);
corr_mat = corrcoef(mat);
%compute RDM matrix
RDM_mat = one_mat - corr_mat;
%plot RDM matrix
figure();
imagesc(RDM_mat);
xticks([4:8:64]);
xticklabels({'Animal(1-8)','Boat(1-8)','Car(1-8)','Chairs(1-8)','Faces(1-8)','Fruits(1-8)','Planes(1-8)','Tables(1-8)'})
xtickangle(320)
yticks([4:8:64]);
yticklabels({'Animal(1-8)','Boat(1-8)','Car(1-8)','Chairs(1-8)','Faces(1-8)','Fruits(1-8)','Planes(1-8)','Tables(1-8)'})
title(["Representational Dissimilarity Matrix for " + titleplot])
colorbar;
axis square
end

