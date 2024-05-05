import glob
import shutil
from pathlib import Path

if __name__ == "__main__":
    last_output_path = sorted(glob.glob("out/sender_receiver/*"))[-1]

    extr_path = Path("extracted")
    for folder in glob.glob(str(last_output_path) + "/*"):
        copy_files_list = glob.glob(str(folder) + "/*.png")
        copy_files_list += glob.glob(str(folder) + "/*.pdf")
        copy_files_list.append(folder + "/images/cells_at_iter_0000000300.png")
        copy_files_list.append(folder + "/images/cells_at_iter_0000036000.png")
        for png_file in copy_files_list:
            e_path = extr_path / Path(folder).name
            e_path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(png_file, e_path / Path(png_file).name)
