import urllib
import os

dest_dir = "./all_images_new/"

def down_from_dir(dirname):
    file_names = os.listdir(dirname)

    for file_name in file_names:
        category_path = os.path.join(dirname, file_name)
        category = file_name[:-4]

        category_dir = open(category_path, 'r')

        for index, image_url in enumerate(category_dir):

            if index % 10 == 0:
                print(index, image_url)

            try:
                uopen = urllib.urlopen(image_url)
                stream = uopen.read()
                filename_new = dest_dir + category + dirname[-6:] + "_" + str(index) + "_.jpg"
                while os.path.isfile(filename_new):
                    index = index + 1
                    filename_new = dest_dir + category + dirname[-6:] + "_" + str(index) + "_.jpg"
                file = open(filename_new, 'w')
                file.write(stream)
                file.close()
            except:
                pass

        category_dir.close()


def down_from_txt(textname):
    category = open(textname, 'r')

    for index, image_url in enumerate(category):
        if index % 10 == 0:
            print(index, image_url)

        try:
            uopen = urllib.urlopen(image_url)
            stream = uopen.read()
            filename_new = dest_dir + "others_file_" + str(index) + "_.jpg"
            while os.path.isfile(filename_new):
                index = index + 1
                filename_new = dest_dir + "others_file_" + str(index) + "_.jpg"
            file = open(filename_new, 'w')
            file.write(stream)
            file.close()

        except:
            pass

    category.close()



if __name__ == "__main__":

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    down_from_dir("./indoor")
    down_from_dir("./outdoor")
    down_from_txt("others.txt")
