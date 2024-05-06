species_group = ['truncata','veitchii','ventricosa']
def create_sql_table(species_group):
    sql_create_database = "CREATE DATABASE IF NOT EXISTS `nepenthes_multilabel_classification` /*!40100 DEFAULT CHARACTER SET utf8 */;\n"

    sql_use_database = "USE `nepenthes_multilabel_classification`;\n"

    sql_statement = "CREATE TABLE IF NOT EXISTS `item_list` (\n"
    sql_statement += "\t`id` int(11) NOT NULL AUTO_INCREMENT,\n"
    sql_statement += "\t`species` varchar(100) NOT NULL,\n"
    sql_statement += "\t`serial` char(7) NOT NULL,\n"
    sql_statement += "\t`source` varchar(50) NOT NULL,\n"

    for species in species_group:
        sql_statement += f"\t`{species}` BOOLEAN NOT NULL,\n"

    sql_statement += "PRIMARY KEY (`id`, `serial`))\n" 
    sql_statement += "ENGINE=InnoDB DEFAULT CHARSET=utf8;\n"
    sql_statement += "CREATE UNIQUE INDEX unique_serial ON `item_list` (`serial`);\n"

    with open('create_table.sql', 'w') as file:
        file.write(sql_create_database)
        file.write(sql_use_database)        
        file.write(sql_statement)

    print("produce create_table.sql")

create_sql_table(species_group)  