import xlrd # module to parse excel files
import numpy 
import json
from subprocess import call
from xml.dom import minidom

def xlParser(excel_file,sheet_name):

    book = open_workbook(excel_file)
    sheet = book.sheet_by_name(sheet_name)
	
	
	for row_num in list(range(0,sheet.nrows)):
		product = str(sheet.cell(row_num,1).value)
		description  = str(sheet.cell(row_num,2).value)
		product_list.append(product)
		description_list.append(description)
		
	data_mat = numpy.array(product_list, description_list)
	
	
	return data_mat
	
def jsonParser(json.file):

	with open(json.file) as data_file:    
    data = json.load(data_file)
	
	# retrieve components as follows
	data["product_name"][0]
	data["description"][0]
	
	#export json file into mongodb as collection
	call(["mongoimport" "--db" "dbName" "--collection collectionName" "--file" "data.json" ]
	return data
	
def xmlParser(xml_file):

	xmldoc = minidom.parse(xml_file)
	itemlist = xmldoc.getElementsByTagName('item')
	print(itemlist[0].attributes['name'].value)
	for s in itemlist:
		print(s.attributes['name'].value)
	

def dataParser():
	
	data_mat = xlParser("Product_Catalog.xls","Mapping"):
	jsonParser('data.json')
	xmlParser(xml_file)


dataParser()