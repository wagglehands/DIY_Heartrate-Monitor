
data=xlsread('ExcelFilename.xlsx','ExcelSheetName');

ECGExtract=(data(1:end,1)); %%Column A data 

fid=fopen('test.txt','wt');%opening with the t flag auto-converts \n to \r\n on Windows
fprintf(fid,'{');
FormatSpec=[repmat('%i ',1,size(ECGExtract,2)) ','];%or should that have been \r\n instead?
fprintf(fid,FormatSpec, ECGExtract);
fprintf(fid,'}')
fclose(fid);