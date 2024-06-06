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

        % conditions folder inside the subfolder
        conditions_dir = fullfile(subfolder_dir, 'conditions');

        % List of .mat files in the conditions folder
        mat_files = dir(fullfile(conditions_dir, '*.mat'));

        % Loop through each .mat file and convert labels to .csv
        for k = 1:length(mat_files)
            mat_file_path = fullfile(conditions_dir, mat_files(k).name);

            % Load the .mat file
            mat_data = load(mat_file_path);

            % Assuming the labels are stored in a variable named 'labels' in the .mat file
            % Modify 'labels' to the appropriate variable name if different
            if isfield(mat_data, 'labels')
                labels = mat_data.labels;

                % Create the output .csv file name
                [~, file_name, ~] = fileparts(mat_files(k).name);
                csv_file_path = fullfile(conditions_dir, [file_name '_labels.csv']);

                % Convert the labels to a table if necessary and write to .csv
                if isstruct(labels)
                    labels = struct2table(labels);
                end
                writetable(labels, csv_file_path);
            else
                warning('No ''labels'' variable found in %s', mat_file_path);
            end
        end
    end
end
