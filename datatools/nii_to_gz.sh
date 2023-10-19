#!/bin/bash

# Путь к корневой директории
root_dir="/path/to/root/directory"

# Рекурсивно находим все файлы с расширением .nii и переименовываем их
find "$root_dir" -type f -name "*.nii" -exec sh -c '
  for file do
    new_name=$(echo "$file" | sed "s/\.nii$/.nii.gz/")
    mv "$file" "$new_name"
    echo "Renamed $file to $new_name"
  done
' sh {} +
