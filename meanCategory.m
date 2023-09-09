function [meanCategory] = meanCategory(it_data,category)


%Calculate mean of IT Neural Responses in each category using a for loop
sumCategory = 0;
for i = 1:5760
    for j = 1:720
        if i == category(j)
            sumCategory = sumCategory + it_data(i);
        end
    end
end

meanCategory = mean(sumCategory);
end

