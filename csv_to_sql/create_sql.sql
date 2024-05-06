CREATE DATABASE IF NOT EXISTS `nepenthes_multilabel_classification` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `nepenthes_multilabel_classification`;

CREATE TABLE IF NOT EXISTS `item_list` (
  `id` int(11) NOT NULL AUTO_INCREMENT,     
  `species` varchar(100) NOT NULL,
  `serial` char(7) NOT NULL,
  `source` varchar(50) NOT NULL,
  `truncata` BOOL NOT NULL,
  `veitchii` BOOL NOT NULL,
  `ventricosa` BOOL NOT NULL,
   PRIMARY KEY (`id`,`serial`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE UNIQUE INDEX unique_serial
ON `item_list` (`serial`);



