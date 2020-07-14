clc;
clear;
for id=1:10
    if (id == 5 || id == 8)
        continue;
    end
    %filename = append('I:/GradEs/Testing/NormGoodBMP/',int2str(id),'/imadjust.csv');
    filename = ['I:/GradEs/Testing/NormGoodBMP/' num2str(id) '/non_imadjust.csv'];
    x = xlsread(filename);
    for j=1:36
        for i=1:10
            x1(i)=x(i,j);
        end
        array = x1;
        MO = mean(array);
        SKO = std(array);
        arrayMO(id, j)= MO;
        arraySKO(id, j)= SKO;
    end
end