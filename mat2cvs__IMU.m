% Base directory containing the Subjects folder
base_dir = 'Subjects'; % 필요한 경로로 변경하세요

% List of subject folders
subject_folders = dir(fullfile(base_dir, 'AB*'));

% Loop through each subject folder
for i = 1:length(subject_folders)
    subject_dir = fullfile(base_dir, subject_folders(i).name);

    % List of subfolders inside each subject folder
    subfolders = {'levelground', 'ramp', 'stair', 'treadmill'};

    for j = 1:length(subfolders)
        subfolder_dir = fullfile(subject_dir, subfolders{j});

        % imu folder inside the subfolder
        imu_dir = fullfile(subfolder_dir, 'imu');

        % List of .mat files in the imu folder
        mat_files = dir(fullfile(imu_dir, '*.mat'));

        % Loop through each .mat file and convert it to .csv
        for k = 1:length(mat_files)
            mat_file_path = fullfile(imu_dir, mat_files(k).name);

            % Load the .mat file
            mat_data = load(mat_file_path);

            % Assuming the data is stored in a variable named 'data' in the .mat file
            % Modify 'data' to the appropriate variable name if different
            data_fieldnames = fieldnames(mat_data);
            data = mat_data.(data_fieldnames{1}); 

            % Create the output .csv file name
            [~, file_name, ~] = fileparts(mat_files(k).name);
            csv_file_path = fullfile(imu_dir, [file_name '.csv']);

            % Convert the data to a table if necessary and write to .csv
            if isstruct(data)
                data = struct2table(data);
            end
            writetable(data, csv_file_path);
        end
    end
end
