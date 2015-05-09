figure
hold on
for i=1:4
    rayPath = zeros(13,3);
    for j=1:7
        if(isnan(rayLog(i,j)))
            break;
        end
        rayPath(j, 1:3) = rayLog(i,j*3-2:j*3);
    end
    plot3(rayPath(:,1),rayPath(:,2),rayPath(:,3));
    xlabel('x');
    ylabel('y');
    zlabel('z');
end
hold off