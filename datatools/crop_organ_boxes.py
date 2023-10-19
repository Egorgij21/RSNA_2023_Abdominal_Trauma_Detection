def flip_3d(image ,axis):
    assert axis in [0, 1, 2], "axis should be in [0, 1, 2]"
    if axis == 0:
        return image[::-1]
    if axis == 1:
        return image[:, ::-1, :]
    return image[:, :, ::-1]

def get_coords(row : pd.DataFrame, image, padding : float):
    x_start, x_end = row.x_min.tolist()[0] * image.shape[2], row.x_max.tolist()[0] * image.shape[2]
    x_start, x_end = max(x_start - ((x_end - x_start) * padding), 0) , min(x_end + ((x_end - x_start) * padding), image.shape[2])
    
    y_start, y_end = row.y_min.tolist()[0] * image.shape[1], row.y_max.tolist()[0] * image.shape[1]
    y_start, y_end = max(y_start - ((y_end - y_start) * padding), 0) , min(y_end + ((y_end - y_start) * padding), image.shape[1])
    
    z_start, z_end = row.z_min.tolist()[0] * image.shape[0], row.z_max.tolist()[0] * image.shape[0]
    z_start, z_end = max(z_start - ((z_end - z_start) * padding), 0) , min(z_end + ((z_end - z_start) * padding), image.shape[0])
    
    return int(x_start), int(x_end), int(y_start), int(y_end), int(z_start), int(z_end)

def get_concat(image, coords_l, coords_r) -> np.array:
    x_start_l, x_end_l, y_start_l, y_end_l, z_start_l, z_end_l = coords_l
    x_start_r, x_end_r, y_start_r, y_end_r, z_start_r, z_end_r = coords_r
    
    max_y = max(y_end_l - y_start_l, y_end_r - y_start_r)
    max_z = max(z_end_l - z_start_l, z_end_r - z_start_r)
    
    cropped_image_l = image[z_start_l : z_end_l, y_start_l : y_end_l, x_start_l : x_end_l]
    cropped_image_r = image[z_start_r : z_end_r, y_start_r : y_end_r, x_start_r : x_end_r]
    
    left_sitk  = sitk.GetImageFromArray(cropped_image_l)
    left_sitk.SetOrigin((0, 0, 0))
    spacing = left_sitk.GetSpacing()
    left_sitk.SetSpacing(spacing)
    left_sitk = resize_sitk(left_sitk, (x_end_l - x_start_l, max_y, max_z))
    
    right_sitk  = sitk.GetImageFromArray(cropped_image_r)
    right_sitk.SetOrigin((0, 0, 0))
    spacing = left_sitk.GetSpacing()
    right_sitk.SetSpacing(spacing)
    right_sitk = resize_sitk(right_sitk, (x_end_r - x_start_r, max_y, max_z))
    
    left = sitk.GetArrayFromImage(left_sitk)
    right = sitk.GetArrayFromImage(right_sitk)
    organ = np.concatenate([left, right], axis=2)

    image_mean = organ.mean()
    image_std = organ.std()
    organ = (organ - image_mean) / image_std
    
    # for i in range(1, 5000, 10):
    #     fig, axes = plt.subplots(ncols=2)
    #     axes[0].imshow(organ[i,:,:])
    #     axes[1].imshow(organ[i,:,:])
    # assert 1 == 0
    
    return organ
    
def read_sitk(folder, ext=".png"):
    paths = glob.glob(f"{folder}/*.png")
    paths_sorted = sorted(paths, key=lambda s: int(s.split("/")[-1].split(".")[0]))


    try:
        out = sitk.ReadImage(paths_sorted)
    except:
        print(folder)

    return out

def resize_sitk(image, output_size, is_mask=False):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_spacing = [
        original_spacing[i] * (original_size[i] / output_size[i]) for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    )

    resized_image = resampler.Execute(image)

    return resized_image

def save_crops(df, loader, images_folder, output_folder, label=[4, 3], padding=0.0) -> None:
    """
    Находит ограничивающие параллелепипеды для каждого класса в предсказании
    и сохраняет их как бинарные 3D-маски и соответствующие им подволюмы из GT-изображения.

    Параметры:
        prediction (numpy.ndarray): 3D-массив с номерами классов.
        gt_image (numpy.ndarray): 3D-массив с исходным изображением.
        output_folder (str): Папка для сохранения результатов.
        padding (float): Отступ для 3D-маски, доля от 0 до 1.
    """
    shapes = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    

    for sample in loader:
        patient_id = sample['patient_id'][0].item()
        scan_id = sample['scan_id'][0].item()
        path_to_image = sample['path']
        
        path_to_save = os.path.join(output_folder, str(patient_id))
        
        # sitk_image = read_sitk(path_to_image)
        gt_image = sample['image'][0]
        
        rows = df[(df.patient_id == patient_id) & (df.scan_id == scan_id)]

        assert len(rows) <= 2
        
        if len(rows) == 2:
            try:
                l_coords = get_coords(rows[rows['class'] == 4], gt_image, padding)
                r_coords = get_coords(rows[rows['class'] == 3], gt_image, padding)
                organ = get_concat(gt_image, l_coords, r_coords)
                shapes.append(organ.shape)
                # Сохраняем как бинарную 3D-маску и подволюм изображения

                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)

                sub_volume_image_sitk = sitk.GetImageFromArray(organ)
                sitk.WriteImage(sub_volume_image_sitk, os.path.join(path_to_save, 'image.nii.gz'))
                
            except:
                print(f'Error: {path_to_save}')
        else:
            print(f"No samples: {sample['patient_id'][0].item()}/{sample['scan_id'][0].item()}")
    return shapes


im_path = 'data/train_images'
save_path = 'data/kidneys_crops_padd'
shapes = save_crops(df, test_loader, im_path, output_folder = save_path, label=[4, 3], padding=0.1)











def findSingleBoundingBoxPerClass(prediction : torch.Tensor, padding : float) -> dict:
    bounding_boxes = {}
    classes = np.unique(prediction)
    z, y, x = prediction.shape
    for c in classes:
        if c == 0:  # фон
            continue
        coords = np.argwhere(prediction == c)
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)

        x_padding = int((x_max - x_min) * padding)
        y_padding = int((y_max - y_min) * padding)
        z_padding = int((z_max - z_min) * padding)

        x_min = max(x_min - x_padding, 0) / x
        y_min = max(y_min - y_padding, 0) / y
        z_min = max(z_min - z_padding, 0) / z
        x_max = min(x_max + x_padding, prediction.shape[2]) / x
        y_max = min(y_max + y_padding, prediction.shape[1]) / y
        z_max = min(z_max + z_padding, prediction.shape[0]) / z

        bounding_boxes[c] = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    
    return bounding_boxes

def getCoords(bounding_boxes : dict, label : int, padding : float) -> Tuple[int]:
    x_start, x_end, y_start, y_end, z_start, z_end = bounding_boxes[label]
    
    x_start, x_end = max(x_start - ((x_end - x_start) * padding), 0) , min(x_end + ((x_end - x_start) * padding), image.shape[2])
    y_start, y_end = max(y_start - ((y_end - y_start) * padding), 0) , min(y_end + ((y_end - y_start) * padding), image.shape[1])
    z_start, z_end = max(z_start - ((z_end - z_start) * padding), 0) , min(z_end + ((z_end - z_start) * padding), image.shape[0])
    
    return int(x_start), int(x_end), int(y_start), int(y_end), int(z_start), int(z_end)

def cropOrgan(full_cube : np.array, coords : Tuple[int]) -> np.array:
    x_start, x_end, y_start, y_end, z_start, z_end = coords
    
    organ = full_cube[z_start : z_end, y_start : y_end, x_start : x_end]
    
    image_mean = organ.mean()
    image_std = organ.std()
    organ = (organ - image_mean) / image_std
    
    return organ

def getOrganCube(full_cube : np.array, bounding_boxes : dict, label : int, padding : float) -> np.array:
    coords = getCoords(bounding_boxes, label, padding)
    return cropOrgan(full_cube, coords)

bounding_boxes = findSingleBoundingBoxPerClass(prediction, padding=0.0)
organ = getOrganCube(cube, bounding_boxes, label=1, padding=0.0)