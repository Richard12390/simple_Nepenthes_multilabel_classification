import csv
import pymysql 

db = pymysql.connect(host="localhost",user="root",password="",database="nepenthes_multilabel_classification",charset="utf8")
cursor = db.cursor() 

csvfile = "images.csv"

def bool_to_int(value):
     return '1' if value == 'True' else '0'

with open(csvfile, 'r', encoding = "utf-8",errors='ignore') as fp:

     reader = csv.reader(fp)
     header=next(reader)

     for row_idx,row in enumerate(reader): 
          for idx, item in enumerate(row):
               row[idx] = item.strip()
               
               if row[idx] == 'True' or row[idx] == 'False':
                    row[idx] = bool_to_int(row[idx])

          sql = """INSERT INTO `item_list`(`{}`)
               VALUES ('{}')
               """.format("`,`".join(header[:]), "','".join(row[:]))
         
          print(sql)


          try:
               cursor.execute(sql) 
               db.commit()
               print("add new data success: ", row)
          except:
               db.rollback()
               print("fail to add new data: ", row)

db.close() 

