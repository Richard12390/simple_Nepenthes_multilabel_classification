import csv
import os 

def rows_dict_to_csv(rows_dict, species_group, csvfile):
    
    print(rows_dict)
    print(species_group)
    print(csvfile)

    sorted_rows_dict = sorted(rows_dict.items(), key=lambda k: k[0])

    if not os.path.exists(csvfile):
        with open(csvfile, 'w', newline='') as fp:
            writer = csv.DictWriter(fp,['species','serial','source']+species_group)
            writer.writeheader()
    with open(csvfile,'r', newline = '') as file_check:
        reader = csv.DictReader(file_check)
        existing_line = [line["serial"] for line in reader]
    with open(csvfile,'a', newline = '') as fp:
        writer = csv.DictWriter(fp,['species','serial','source']+species_group)
        for image_path, data in sorted_rows_dict:
            if data["serial"] not in existing_line:
                writer.writerow(data)
