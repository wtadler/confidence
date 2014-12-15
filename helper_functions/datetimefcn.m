function datestr = datetimefcn
datetime = clock;
datestr=num2str(datetime(1)); % year

for i=2:6 % for each datetime element, add it to the string, prefixing a 0 if the element is < 10.
    if datetime(i) >= 10
        datestr = [datestr num2str(round(datetime(i)))];
    elseif datetime(i) < 10
        datestr = [datestr '0' num2str(round(datetime(i)))];
    end
    while i==3;
        datestr = [datestr '_'];
        break
    end
end