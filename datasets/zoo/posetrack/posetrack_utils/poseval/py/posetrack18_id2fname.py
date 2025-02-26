import os

posetrack17_train_sequences = set(
  [
    (1, 8838),
    (1, 12218),
    (1, 6852),
    (1, 16530),
    (1, 12507),
    (1, 14073),
    (1, 9488),
    (1, 22683),
    (1, 16637),
    (1, 7861),
    (1, 8968),
    (1, 43),
    (1, 8732),
    (1, 13627),
    (1, 7380),
    (1, 13780),
    (1, 2716),
    (1, 98),
    (1, 436),
    (1, 14265),
    (1, 17133),
    (1, 16464),
    (1, 9922),
    (1, 10773),
    (1, 7607),
    (1, 228),
    (1, 20924),
    (1, 24635),
    (1, 16571),
    (1, 760),
    (1, 14321),
    (1, 16165),
    (1, 8808),
    (1, 23492),
    (1, 866),
    (1, 6265),
    (1, 16882),
    (1, 275),
    (1, 24985),
    (1, 2905),
    (1, 20928),
    (1, 7851),
    (1, 3402),
    (1, 16171),
    (1, 15882),
    (1, 823),
    (1, 3498),
    (1, 14344),
    (1, 14354),
    (1, 20900),
    (1, 9533),
    (2, 10),
    (1, 14480),
    (1, 15892),
    (1, 3701),
    (1, 15124),
    (1, 16411),
    (1, 9043),
    (1, 9012),
    (1, 8743),
    (1, 12620),
    (2, 28),
    (1, 16440),
    (1, 7855),
    (1, 15130),
    (1, 271),
    (1, 8820),
    (1, 15875),
    (1, 23471),
    (1, 8882),
    (1, 2357),
    (1, 24180),
    (2, 15),
    (1, 23695),
    (1, 16883),
    (1, 231),
    (1, 15290),
    (1, 13337),
    (1, 9003),
    (1, 13908),
    (1, 3403),
    (2, 1),
    (1, 502),
    (1, 9495),
    (1, 14268),
    (1, 8961),
    (1, 15277),
    (1, 8616),
    (1, 14345),
    (1, 14278),
    (1, 985),
    (1, 1243),
    (1, 11989),
    (1, 15125),
    (1, 13515),
    (2, 29),
    (1, 2839),
    (1, 15366),
    (1, 13821),
    (1, 9718),
    (1, 12056),
    (1, 14052),
    (1, 21077),
    (1, 12268),
    (1, 5728),
    (1, 21133),
    (1, 8819),
    (1, 13965),
    (1, 5759),
    (1, 17180),
    (1, 14553),
    (1, 9506),
    (1, 22671),
    (1, 5847),
    (1, 10288),
    (1, 22682),
    (1, 14231),
    (1, 8969),
    (1, 14403),
    (1, 20896),
    (1, 7381),
    (1, 14375),
    (1, 14122),
    (1, 16535),
    (1, 1158),
    (1, 14506),
    (1, 9598),
    (1, 14334),
    (1, 17184),
    (1, 2893),
    (1, 23699),
    (1, 10010),
    (1, 3730),
    (1, 12023),
    (2, 48),
    (1, 23454),
    (1, 9499),
    (1, 9654),
    (1, 14235),
    (1, 10111),
    (1, 13795),
    (1, 16496),
    (1, 16313),
    (1, 8795),
    (1, 12732),
    (1, 9534),
    (2, 17),
    (1, 439),
    (1, 8744),
    (1, 1682),
    (1, 921),
    (1, 8833),
    (1, 14054),
    (1, 4902),
    (1, 24893),
    (2, 3),
    (1, 10715),
    (1, 15309),
    (1, 820),
    (1, 10542),
    (1, 285),
    (1, 7467),
    (1, 13271),
    (1, 15406),
    (1, 9487),
    (1, 14763),
    (1, 12155),
    (1, 9398),
    (1, 1686),
    (1, 2255),
    (1, 8837),
    (1, 2787),
    (1, 12911),
    (1, 9054),
    (1, 223),
    (1, 14662),
    (1, 12722),
    (2, 27),
    (1, 15585),
    (1, 4833),
    (1, 14551),
    (1, 9504),
    (1, 9555),
    (1, 13787),
    (1, 9993),
    (1, 14363),
    (1, 8803),
    (1, 9411),
    (1, 15189),
    (1, 9617),
    (1, 8725),
    (1, 4891),
    (1, 14390),
    (1, 20822),
    (1, 5732),
    (1, 12859),
    (1, 474),
    (1, 2552),
    (1, 10774),
    (1, 14367),
    (2, 23),
    (1, 520),
    (1, 14272),
    (1, 13527),
    (1, 2234),
    (1, 5841),
    (1, 15899),
    (1, 9538),
    (1, 14266),
    (1, 15537),
    (1, 13671),
    (1, 7387),
    (1, 8906),
    (2, 26),
    (1, 10198),
    (1, 7413),
    (1, 2258),
    (1, 14082),
    (1, 16215),
    (1, 15314),
    (1, 12273),
    (1, 17121),
    (1, 20823),
    (1, 14183),
    (1, 15765),
    (1, 11280),
    (1, 16433),
    (1, 23416),
    (1, 8730),
    (1, 21078),
    (1, 352),
    (1, 17197),
    (1, 14121),
    (1, 22676),
    (1, 14505),
    (1, 14280),
    (1, 4836),
    (2, 22),
    (1, 16198),
    (1, 7392),
    (1, 9445),
    (1, 13268),
    (1, 16211),
    (1, 2264),
    (1, 24487),
    (1, 14698),
    (1, 10007),
    (1, 8812),
    (2, 36),
    (1, 8796),
    (1, 16330),
    (1, 9938),
    (1, 96),
    (1, 17129),
    (1, 8976),
    (1, 22642),
    (1, 7536),
    (1, 385),
    (1, 16417),
    (1, 9597),
    (1, 13512),
    (1, 1157),
    (1, 15293),
    (1, 14178),
    (1, 13557),
    (1, 14129),
    (1, 1491),
    (1, 8877),
    (1, 23484),
    (2, 2),
    (1, 10177),
    (1, 10863),
    (1, 8884),
    (1, 8962),
    (1, 5061),
    (1, 2843),
    (1, 9727),
    (1, 24642),
    (1, 14288),
    (1, 1153),
    (1, 15832),
    (1, 1687),
    (1, 2254),
    (1, 15177),
    (1, 2786),
    (1, 12910),
    (1, 22124),
    (1, 1341),
    (1, 16668),
    (1, 14342),
    (1, 5232),
    (1, 799),
  ]
)

posetrack17_testval_sequences = set(
  [
    (1, 707),
    (1, 11878),
    (1, 16842),
    (1, 5368),
    (1, 286),
    (1, 9528),
    (1, 8993),
    (1, 9038),
    (1, 9468),
    (1, 10127),
    (1, 18900),
    (1, 16611),
    (1, 14361),
    (1, 46),
    (1, 15869),
    (1, 1110),
    (1, 23966),
    (1, 475),
    (1, 18906),
    (1, 20856),
    (1, 229),
    (1, 3136),
    (1, 13029),
    (1, 1757),
    (1, 24566),
    (1, 24153),
    (1, 16451),
    (1, 2838),
    (1, 2266),
    (1, 903),
    (1, 14703),
    (1, 17496),
    (1, 15755),
    (1, 14027),
    (1, 750),
    (1, 3745),
    (3, 2),
    (1, 16235),
    (1, 12967),
    (1, 10130),
    (1, 15294),
    (1, 17447),
    (1, 16517),
    (1, 15521),
    (1, 13601),
    (1, 14376),
    (1, 15149),
    (1, 18719),
    (1, 14313),
    (1, 1970),
    (1, 8894),
    (1, 14292),
    (1, 10309),
    (1, 19980),
    (1, 6503),
    (1, 3504),
    (1, 9472),
    (1, 8826),
    (1, 24177),
    (1, 17434),
    (3, 3),
    (1, 2061),
    (1, 24493),
    (1, 6545),
    (1, 3542),
    (1, 24906),
    (1, 9268),
    (1, 18592),
    (1, 9469),
    (1, 17955),
    (1, 21082),
    (1, 22831),
    (1, 21130),
    (1, 2284),
    (1, 808),
    (1, 15868),
    (1, 21084),
    (1, 12046),
    (1, 1733),
    (1, 24149),
    (1, 12332),
    (1, 17984),
    (1, 11526),
    (1, 2928),
    (1, 5803),
    (1, 23411),
    (1, 15941),
    (1, 2777),
    (1, 16556),
    (1, 9301),
    (1, 23746),
    (1, 18159),
    (1, 10303),
    (1, 9523),
    (1, 22892),
    (1, 10521),
    (1, 18626),
    (1, 7504),
    (1, 18412),
    (1, 1535),
    (1, 14309),
    (1, 1280),
    (1, 15862),
    (1, 2367),
    (1, 22656),
    (1, 3397),
    (1, 14524),
    (1, 18657),
    (1, 9452),
    (1, 8991),
    (1, 5413),
    (1, 3223),
    (1, 9509),
    (1, 8736),
    (1, 10357),
    (1, 20912),
    (1, 161),
    (1, 18296),
    (1, 44),
    (1, 2281),
    (1, 20909),
    (1, 7269),
    (1, 16421),
    (1, 22693),
    (3, 1),
    (1, 14522),
    (1, 15375),
    (1, 24564),
    (1, 1940),
    (1, 14297),
    (1, 19078),
    (1, 15908),
    (1, 16419),
    (1, 9477),
    (1, 2273),
    (1, 7952),
    (1, 24573),
    (1, 9460),
    (3, 5),
    (1, 16576),
    (1, 14317),
    (1, 11287),
    (1, 16194),
    (1, 7681),
    (1, 9458),
    (1, 12838),
    (1, 5799),
    (1, 18623),
    (1, 8761),
    (1, 24516),
    (1, 8160),
    (1, 9526),
    (1, 15859),
    (1, 20818),
    (1, 9403),
    (1, 2279),
    (1, 3416),
    (1, 202),
    (1, 20820),
    (1, 22699),
    (1, 24156),
    (1, 1545),
    (1, 23730),
    (1, 5336),
    (1, 1242),
    (1, 693),
    (1, 14307),
    (1, 15812),
    (3, 4),
    (1, 9602),
    (1, 23444),
    (1, 6818),
    (1, 8847),
    (1, 21086),
    (1, 2286),
    (1, 10517),
    (1, 3546),
    (1, 23965),
    (1, 23736),
    (1, 2852),
    (1, 10350),
    (1, 536),
    (1, 9476),
    (1, 811),
    (1, 3224),
    (1, 83),
    (1, 24876),
    (1, 9404),
    (1, 9521),
    (1, 23719),
    (1, 7500),
    (1, 20819),
    (1, 9527),
    (1, 13602),
    (1, 1282),
    (1, 21123),
    (1, 15278),
    (1, 8789),
    (1, 1537),
    (1, 5592),
    (1, 13534),
    (1, 15302),
    (1, 24158),
    (1, 24621),
    (1, 7684),
    (1, 3742),
    (1, 16662),
    (1, 2276),
    (1, 1735),
    (1, 2835),
    (1, 16180),
    (1, 23717),
    (1, 20880),
    (1, 522),
    (1, 14102),
    (1, 14384),
    (1, 1001),
    (1, 1486),
    (1, 4622),
    (1, 14531),
    (1, 20910),
    (1, 8827),
    (1, 2277),
    (1, 14293),
    (1, 9883),
    (1, 16239),
    (1, 16236),
    (1, 8760),
    (1, 15860),
    (1, 7128),
    (1, 5833),
    (1, 23653),
    (1, 5067),
    (1, 14523),
    (1, 24165),
    (1, 18725),
    (1, 7496),
    (1, 342),
    (1, 17839),
    (1, 15301),
    (1, 24575),
    (1, 2364),
    (1, 1744),
    (1, 13293),
    (1, 14960),
    (1, 22430),
    (1, 23754),
    (1, 3943),
    (1, 12834),
    (1, 22688),
  ]
)

posetrack18_train_sequences = set(
  [
    (1, 15394),
    (1, 16418),
    (1, 20824),
    (1, 12949),
    (1, 24218),
    (1, 14680),
    (1, 9412),
    (1, 14193),
    (1, 11347),
    (1, 16886),
    (1, 15883),
    (1, 1278),
    (1, 12255),
    (1, 23938),
    (1, 23500),
    (1, 18647),
    (1, 5231),
    (1, 22661),
    (1, 24767),
    (1, 24630),
    (1, 22128),
    (1, 15889),
    (1, 24211),
    (1, 17161),
    (1, 24317),
    (1, 24170),
    (1, 23773),
    (1, 13967),
    (1, 9484),
    (1, 1502),
    (1, 18993),
    (1, 23493),
    (1, 20872),
    (1, 24102),
    (1, 18987),
    (1, 20898),
    (1, 18146),
    (1, 11992),
    (1, 17563),
    (1, 23414),
    (1, 16560),
    (1, 14695),
    (1, 10008),
    (1, 5180),
    (1, 17127),
    (1, 10013),
    (1, 13743),
    (1, 8830),
    (1, 24212),
    (1, 7519),
    (1, 12789),
    (1, 5643),
    (1, 13526),
    (1, 23469),
    (1, 18651),
    (1, 24712),
    (1, 12783),
    (1, 14195),
    (1, 14694),
    (1, 16169),
    (1, 9628),
    (1, 9287),
    (1, 20447),
    (1, 12784),
    (1, 10179),
    (1, 22679),
    (1, 9485),
    (1, 14343),
    (1, 18542),
    (1, 1156),
    (1, 17427),
    (1, 22665),
    (1, 16252),
    (1, 8660),
    (1, 24479),
    (1, 5173),
    (1, 10006),
    (1, 19063),
    (1, 24634),
    (1, 20382),
    (1, 17179),
    (1, 20942),
    (1, 10161),
    (1, 13924),
    (1, 14953),
    (1, 220),
    (1, 23882),
    (1, 19998),
    (1, 12980),
    (1, 22911),
    (1, 20258),
    (1, 10486),
    (1, 10005),
    (1, 16461),
    (1, 15022),
    (1, 9615),
    (1, 232),
    (1, 23814),
    (1, 15305),
    (1, 23360),
    (1, 21524),
    (1, 24706),
    (1, 23864),
    (1, 16409),
    (1, 24018),
    (1, 24627),
    (1, 20935),
    (1, 22410),
    (1, 14081),
    (1, 17908),
    (1, 24216),
    (1, 20091),
    (1, 15925),
    (1, 10608),
    (1, 14406),
    (1, 24073),
    (1, 13564),
    (1, 12725),
    (1, 24696),
    (1, 24099),
    (1, 8912),
    (1, 14767),
    (1, 13523),
    (1, 24553),
    (1, 20669),
    (1, 1888),
    (1, 24628),
    (1, 8664),
    (1, 24182),
    (1, 17132),
    (1, 343),
    (1, 15821),
    (1, 23472),
    (1, 21115),
    (1, 9427),
    (1, 21126),
    (1, 14665),
    (1, 20672),
    (1, 12984),
    (1, 18390),
    (1, 17234),
    (1, 5222),
    (1, 8745),
    (1, 16540),
    (1, 16199),
    (1, 23818),
    (1, 16166),
    (1, 23412),
    (1, 24606),
    (1, 13207),
    (1, 14064),
    (1, 23739),
    (1, 24143),
    (1, 10003),
    (1, 13193),
    (1, 14795),
    (1, 16706),
    (1, 13816),
    (1, 18649),
    (1, 24435),
    (1, 10270),
    (1, 22071),
    (1, 13518),
    (1, 15025),
    (1, 23776),
    (1, 8875),
    (1, 23988),
    (1, 1151),
    (1, 15619),
    (1, 12729),
    (1, 11991),
    (1, 23706),
    (1, 23950),
    (1, 13742),
    (1, 11557),
    (1, 20486),
    (1, 7389),
    (1, 24459),
    (1, 24477),
    (1, 16545),
    (1, 10004),
    (1, 1503),
    (1, 10869),
    (1, 12049),
    (1, 12003),
    (1, 11606),
    (1, 20940),
    (1, 20090),
    (1, 15405),
    (1, 13957),
    (1, 486),
    (1, 24489),
    (1, 13511),
    (1, 8842),
    (1, 20256),
    (1, 10484),
    (1, 20547),
    (1, 15926),
    (1, 12981),
    (1, 23949),
    (1, 4334),
    (1, 23812),
    (1, 23503),
    (1, 20008),
    (1, 17405),
    (1, 24704),
    (1, 16473),
    (1, 24258),
    (1, 12182),
    (1, 13047),
    (1, 24179),
    (1, 22876),
    (1, 24222),
    (1, 15893),
    (1, 9432),
    (1, 8835),
    (1, 11351),
    (1, 18096),
    (1, 24644),
    (1, 23942),
    (1, 9289),
    (1, 12860),
    (1, 23488),
    (1, 12790),
    (1, 9055),
    (1, 8918),
    (1, 22957),
    (1, 23694),
    (1, 14319),
    (1, 16563),
    (1, 15260),
    (1, 14346),
    (1, 20006),
    (1, 12766),
    (1, 13940),
    (1, 23170),
    (1, 24344),
    (1, 13763),
    (1, 12921),
    (1, 18071),
    (1, 17165),
    (1, 23024),
    (1, 14267),
    (1, 24174),
    (1, 23172),
    (1, 17191),
    (1, 20573),
    (1, 17999),
    (1, 14901),
    (1, 15915),
    (1, 9494),
    (1, 13273),
    (1, 13549),
    (1, 14286),
    (1, 24722),
    (1, 16197),
    (1, 19591),
    (1, 23233),
    (1, 16255),
    (1, 12022),
    (1, 15949),
    (1, 10001),
    (1, 15265),
    (1, 15897),
    (1, 13516),
    (1, 15760),
    (1, 24817),
    (1, 21871),
    (1, 10312),
    (1, 9446),
    (1, 24788),
    (1, 1886),
    (1, 9941),
    (1, 22669),
    (1, 16240),
    (1, 1339),
    (1, 21073),
    (1, 13201),
    (1, 24872),
    (1, 21067),
    (1, 24638),
    (1, 24793),
    (1, 24184),
    (1, 15395),
    (1, 22873),
    (1, 23482),
    (1, 15890),
    (1, 12936),
    (1, 14707),
    (1, 12701),
    (1, 18109),
    (1, 20894),
    (1, 24641),
    (1, 18392),
    (1, 15478),
    (1, 9498),
    (1, 23501),
    (1, 10983),
    (1, 24110),
    (1, 23364),
    (1, 689),
    (1, 14322),
    (1, 12422),
    (1, 14762),
    (1, 16413),
    (1, 13451),
    (1, 13182),
    (1, 289),
    (1, 13045),
    (1, 13195),
    (1, 21068),
    (1, 24631),
    (1, 14074),
    (1, 22414),
    (1, 23475),
    (1, 21094),
    (1, 24220),
    (1, 17162),
    (1, 22432),
    (1, 24171),
    (1, 23774),
    (1, 19938),
    (1, 11349),
    (1, 10741),
    (1, 10330),
    (1, 23940),
    (1, 4325),
    (1, 9491),
    (1, 13254),
    (1, 15911),
    (1, 14550),
    (1, 14283),
    (1, 22687),
    (1, 11993),
    (1, 17102),
    (1, 16561),
    (1, 16228),
    (1, 13202),
    (1, 8773),
    (1, 20488),
    (1, 24138),
    (1, 14507),
    (1, 10014),
    (1, 23476),
    (1, 24610),
    (1, 24213),
    (1, 17155),
    (1, 23030),
    (1, 24172),
    (1, 12954),
    (1, 14128),
    (1, 23470),
    (1, 13006),
    (1, 3727),
    (1, 13513),
    (1, 16861),
    (1, 16170),
    (1, 15773),
    (1, 24225),
    (1, 16501),
    (1, 17635),
    (1, 14284),
    (1, 9655),
    (1, 21626),
    (1, 22672),
    (1, 14039),
    (1, 22948),
    (1, 23219),
    (1, 13952),
    (1, 23977),
    (1, 22666),
    (1, 12020),
    (1, 23394),
    (1, 24131),
    (1, 5174),
    (1, 15772),
    (1, 13197),
    (1, 10864),
    (1, 12920),
    (1, 24074),
    (1, 21139),
    (1, 13820),
    (1, 14557),
    (1, 20096),
    (1, 20825),
    (1, 15029),
    (1, 16163),
    (1, 23958),
    (1, 18679),
    (1, 13264),
    (1, 14747),
    (1, 16462),
    (1, 15921),
    (1, 9444),
    (1, 10514),
    (1, 14277),
    (1, 22099),
    (1, 24192),
    (1, 1884),
    (1, 24173),
    (1, 15306),
    (1, 20925),
    (1, 20011),
    (1, 12596),
    (1, 14335),
    (1, 15778),
    (1, 728),
    (1, 17051),
    (1, 16246),
    (1, 23865),
    (1, 24636),
    (1, 14055),
    (1, 17181),
    (1, 22411),
    (1, 24190),
    (1, 2262),
    (1, 24217),
    (1, 10157),
    (1, 23771),
    (1, 16164),
    (1, 19951),
    (1, 2906),
    (1, 13670),
    (1, 18844),
    (1, 5389),
    (1, 13834),
    (1, 23805),
    (1, 2720),
    (1, 9845),
    (1, 9050),
    (1, 14320),
    (1, 24632),
    (1, 24708),
    (1, 9558),
    (1, 13003),
    (1, 1889),
    (1, 15828),
    (1, 24629),
    (1, 14072),
    (1, 23715),
    (1, 24183),
    (1, 15822),
    (1, 23473),
    (1, 24210),
    (1, 13637),
    (1, 24486),
    (1, 24169),
    (1, 23772),
    (1, 12935),
    (1, 21984),
    (1, 15275),
    (1, 12830),
    (1, 9599),
    (1, 17186),
    (1, 12985),
    (1, 8752),
    (1, 13252),
    (1, 114),
    (1, 11999),
    (1, 23413),
    (1, 8908),
    (1, 23853),
    (1, 13200),
    (1, 23716),
    (1, 14804),
    (1, 14108),
    (1, 13261),
    (1, 23165),
    (1, 24608),
    (1, 7518),
    (1, 12788),
    (1, 24699),
    (1, 18650),
    (1, 3440),
    (1, 16465),
    (1, 19972),
    (1, 13519),
    (1, 20837),
    (1, 23955),
    (1, 9286),
    (1, 16636),
    (1, 18087),
    (1, 8885),
    (1, 22912),
    (1, 23777),
    (1, 13140),
    (1, 14274),
    (1, 13749),
    (1, 23820),
    (1, 23707),
    (1, 15791),
    (1, 10298),
    (1, 22664),
    (1, 16243),
    (1, 20487),
    (1, 24710),
    (1, 16683),
    (1, 24478),
    (1, 18594),
    (1, 12053),
    (1, 24633),
    (1, 9559),
    (1, 14797),
    (1, 10160),
    (1, 24187),
    (1, 12047),
    (1, 15269),
    (1, 21096),
    (1, 16595),
    (1, 13958),
    (1, 24797),
    (1, 19997),
    (1, 22910),
    (1, 15764),
    (1, 22937),
    (1, 15927),
    (1, 23422),
    (1, 20443),
    (1, 9614),
    (1, 16932),
    (1, 23813),
    (1, 23708),
    (1, 9613),
    (1, 10847),
    (1, 7875),
    (1, 18652),
    (1, 12183),
    (1, 24626),
    (1, 15528),
    (1, 24457),
    (1, 13948),
    (1, 349),
    (1, 24320),
    (1, 15311),
    (1, 8801),
    (1, 2260),
    (1, 12940),
    (1, 19949),
    (1, 8731),
    (1, 18097),
    (1, 13249),
    (1, 24098),
    (1, 5228),
    (1, 14766),
    (1, 7876),
    (1, 20456),
    (1, 24147),
    (1, 17097),
    (1, 24822),
    (1, 24318),
    (1, 23713),
    (1, 10114),
    (1, 24181),
    (1, 10009),
    (1, 19066),
    (1, 24208),
    (1, 22127),
    (1, 13522),
    (1, 14664),
    (1, 15283),
    (1, 10317),
    (1, 8806),
    (1, 13825),
    (1, 17652),
    (1, 10197),
    (1, 2229),
    (1, 13274),
    (1, 23495),
    (1, 13550),
    (1, 16248),
    (1, 22675),
    (1, 15356),
    (1, 1152),
    (1, 24480),
    (1, 16565),
    (1, 14200),
    (1, 13206),
    (1, 24588),
    (1, 14071),
    (1, 17594),
    (1, 17261),
    (1, 22094),
    (1, 14511),
    (1, 12050),
    (1, 23738),
    (1, 13192),
    (1, 14374),
    (1, 18153),
    (1, 10321),
    (1, 12746),
    (1, 17054),
    (1, 18648),
    (1, 23926),
    (1, 9421),
    (1, 8880),
    (1, 14715),
    (1, 24818),
    (1, 21864),
    (1, 15315),
    (1, 24649),
    (1, 9447),
    (1, 10002),
    (1, 20182),
    (1, 24570),
    (1, 13741),
    (1, 731),
    (1, 22670),
    (1, 24433),
    (1, 21074),
    (1, 24873),
    (1, 10140),
    (1, 13790),
    (1, 10868),
    (1, 24639),
    (1, 12002),
    (1, 17176),
    (1, 10166),
    (1, 24185),
    (1, 15404),
    (1, 22874),
    (1, 10862),
    (1, 23887),
    (1, 14794),
    (1, 15891),
    (1, 8841),
    (1, 22908),
    (1, 14751),
    (1, 14402),
    (1, 24117),
    (1, 13560),
    (1, 24583),
    (1, 24805),
    (1, 23502),
    (1, 14323),
    (1, 22663),
    (1, 16414),
    (1, 13046),
    (1, 21069),
    (1, 16432),
    (1, 20007),
    (1, 17169),
    (1, 20932),
    (1, 3698),
    (1, 24009),
    (1, 22662),
    (1, 21095),
    (1, 9536),
    (1, 20385),
    (1, 23775),
    (1, 8834),
    (1, 20853),
    (1, 23638),
    (1, 14873),
    (1, 9594),
    (1, 13838),
    (1, 23941),
    (1, 9288),
    (1, 9500),
    (1, 13517),
    (1, 929),
    (1, 14764),
    (1, 23693),
    (1, 17264),
    (1, 14791),
    (1, 24145),
    (1, 9220),
    (1, 15251),
    (1, 24585),
    (1, 14076),
    (1, 17021),
    (1, 17258),
    (1, 12765),
    (1, 20512),
    (1, 19064),
    (1, 24214),
    (1, 8972),
    (1, 16314),
    (1, 11614),
    (1, 13520),
    (1, 14199),
    (1, 14057),
    (1, 11329),
    (1, 10495),
    (1, 24226),
    (1, 9493),
    (1, 13272),
    (1, 14723),
    (1, 2903),
    (1, 14285),
    (1, 12392),
    (1, 9939),
    (1, 23232),
    (1, 16254),
    (1, 9505),
    (1, 8799),
    (1, 14338),
    (1, 17050),
    (1, 8662),
    (1, 5175),
    (1, 22701),
    (1, 15895),
    (1, 8817),
    (1, 14663),
    (1, 9994),
    (1, 7540),
    (1, 10934),
    (1, 16172),
    (1, 5819),
    (1, 7534),
    (1, 23505),
    (1, 20868),
    (1, 18511),
    (1, 8872),
    (1, 18653),
    (1, 24810),
    (1, 13014),
    (1, 11520),
    (1, 15307),
    (1, 23711),
    (1, 16442),
    (1, 22668),
    (1, 16247),
    (1, 21072),
    (1, 9060),
    (1, 15763),
    (1, 24646),
    (1, 10164),
    (1, 24191),
  ]
)

posetrack18_testval_sequences = set(
  [
    (1, 9475),
    (1, 14326),
    (1, 3738),
    (1, 15241),
    (1, 24593),
    (1, 23748),
    (1, 7973),
    (1, 14312),
    (1, 7686),
    (1, 9508),
    (1, 14301),
    (1, 10230),
    (1, 2059),
    (1, 7934),
    (1, 24329),
    (1, 18911),
    (1, 20909),
    (1, 21785),
    (1, 5068),
    (1, 3511),
    (1, 22650),
    (1, 45),
    (1, 3419),
    (1, 7950),
    (1, 20915),
    (1, 18913),
    (1, 14546),
    (1, 2376),
    (1, 691),
    (1, 12201),
    (1, 10992),
    (1, 14535),
    (1, 15498),
    (1, 23933),
    (1, 10016),
    (1, 23646),
    (1, 24577),
    (1, 756),
    (1, 809),
    (1, 16578),
    (1, 18092),
    (1, 16443),
    (1, 3310),
    (1, 5827),
    (1, 12331),
    (1, 14437),
    (1, 17448),
    (1, 3219),
    (1, 2374),
    (1, 18628),
    (1, 7676),
    (1, 2269),
    (1, 21120),
    (1, 24907),
    (1, 2366),
    (1, 14305),
    (1, 14530),
    (1, 9080),
    (1, 803),
    (1, 20374),
    (1, 15372),
    (1, 2060),
    (1, 16677),
    (1, 698),
    (1, 14302),
    (1, 18896),
    (1, 23963),
    (1, 16658),
    (1, 17446),
    (1, 3420),
    (1, 9479),
    (1, 5066),
    (1, 1395),
    (1, 4929),
    (1, 22836),
    (1, 1240),
    (1, 3203),
    (1, 11638),
    (1, 14316),
    (1, 1111),
    (1, 14300),
    (1, 5296),
    (1, 23931),
    (1, 814),
    (1, 10129),
    (1, 4338),
    (1, 18627),
    (1, 21065),
    (1, 5290),
    (1, 3503),
    (1, 21118),
    (1, 12833),
    (1, 863),
    (1, 7938),
    (1, 24338),
    (1, 4707),
    (1, 5420),
    (1, 16204),
    (1, 8901),
    (1, 14330),
    (1, 9470),
    (1, 2267),
    (1, 804),
    (1, 1007),
    (1, 23752),
    (1, 2227),
    (1, 21862),
    (1, 20777),
    (1, 16455),
    (1, 806),
    (1, 9609),
    (1, 2271),
    (1, 18910),
    (1, 752),
    (1, 20031),
    (1, 2274),
    (1, 18712),
    (1, 19395),
    (1, 6807),
    (1, 3418),
    (1, 3950),
    (1, 14545),
    (1, 19076),
    (1, 9039),
    (1, 15845),
    (1, 5830),
    (1, 23932),
    (1, 807),
    (1, 14314),
    (1, 24157),
    (1, 18713),
    (1, 9454),
    (1, 22691),
    (1, 6540),
    (1, 2277),
    (1, 2285),
    (1, 16823),
    (1, 1969),
    (1, 6703),
    (1, 2373),
    (1, 9520),
    (1, 14611),
    (1, 2246),
    (1, 1588),
    (1, 24199),
    (1, 15690),
    (1, 18630),
    (1, 14304),
    (1, 583),
    (1, 14529),
    (1, 23732),
    (1, 15753),
    (1, 802),
    (1, 758),
    (1, 24624),
    (1, 14295),
    (1, 23962),
    (1, 24334),
    (1, 7216),
    (1, 14296),
    (1, 20420),
    (1, 9478),
    (1, 6820),
    (1, 17450),
    (1, 1022),
    (1, 735),
    (1, 3445),
    (1, 18915),
    (1, 754),
    (1, 15239),
    (1, 21083),
    (1, 15374),
    (1, 753),
    (1, 707),
    (1, 18898),
    (1, 901),
    (1, 15944),
    (1, 7219),
    (1, 21788),
    (1, 24007),
    (1, 6538),
    (1, 22653),
    (1, 7728),
    (1, 2283),
    (1, 23390),
    (1, 9473),
    (1, 1634),
    (1, 2371),
    (1, 10524),
    (1, 4524),
    (1, 18061),
    (1, 9531),
    (1, 6510),
    (1, 4815),
    (1, 12045),
    (1, 20776),
    (1, 3747),
    (1, 16234),
    (1, 18909),
    (1, 11648),
    (1, 737),
    (1, 5337),
    (1, 24514),
    (1, 9653),
    (1, 15907),
    (1, 3417),
    (1, 24332),
    (1, 15863),
    (1, 862),
    (1, 22833),
    (1, 20911),
    (1, 2245),
    (1, 24200),
    (1, 12147),
    (1, 7974),
    (1, 9460),
    (1, 9453),
    (1, 23754),
    (1, 1323),
    (1, 3124),
    (1, 9471),
    (1, 24341),
    (1, 5592),
    (1, 630),
    (1, 21561),
    (1, 15084),
    (1, 18657),
    (1, 18381),
    (1, 10516),
    (1, 22651),
    (1, 18090),
    (1, 4712),
    (1, 16195),
    (1, 9532),
    (1, 13287),
    (1, 1962),
    (1, 17437),
    (1, 23744),
    (1, 23934),
    (1, 10529),
    (1, 24159),
    (1, 2),
    (1, 757),
    (1, 14520),
    (1, 23510),
    (1, 5292),
    (1, 23749),
    (1, 5370),
    (1, 24336),
    (1, 1934),
    (1, 7793),
    (1, 10017),
    (1, 24499),
    (1, 24617),
    (1, 2375),
    (1, 23718),
    (1, 1417),
    (1, 2270),
    (1, 20609),
    (1, 8789),
    (1, 2243),
    (1, 22905),
    (1, 812),
    (1, 5365),
    (1, 24154),
    (1, 4621),
    (1, 15088),
    (1, 749),
    (1, 16238),
    (1, 4687),
    (1, 9266),
    (1, 21066),
    (1, 12968),
    (1, 14099),
    (1, 6537),
    (1, 21116),
    (1, 16423),
    (1, 2282),
    (1, 23649),
    (1, 2772),
    (1, 17212),
    (1, 20502),
    (1, 1241),
    (1, 14140),
    (1, 9079),
    (1, 22842),
    (1, 14317),
    (1, 24616),
    (1, 18625),
    (1, 20882),
    (1, 9459),
    (1, 2214),
    (1, 14772),
    (1, 18908),
    (1, 9607),
    (1, 15946),
    (1, 15933),
    (1, 18903),
    (1, 14035),
    (1, 3508),
    (1, 19583),
  ]
)


def idx2seqtype(idx):
  if idx == 1:
    return "mpii"
  elif idx == 2:
    return "bonn"
  elif idx == 3:
    return "mpiinew"
  else:
    assert False


def seqtype2idx(seqtype):
  if seqtype == "mpii":
    return 1
  elif seqtype == "bonn":
    return 2
  elif seqtype in ["mpiinew"]:
    return 3
  else:
    print("unknown sequence type:", seqtype)
    assert False


def posetrack18_id2fname(image_id):
  """Generates filename given image id 

  Args:
    id: integer in the format TSSSSSSFFFF, 
      T encodes the sequence source (1: 'mpii', 2: 'bonn', 3: 'mpiinew')
      SSSSSS is 6-digit index of the sequence
      FFFF is 4-digit index of the image frame 
      
  Returns:
    name of the video sequence
  """
  seqtype_idx = image_id // 10000000000
  seqidx = (image_id % 10000000000) // 10000
  frameidx = image_id % 10000

  fname = "{:06}_{}".format(seqidx, idx2seqtype(seqtype_idx))

  if (seqtype_idx, seqidx) in posetrack17_testval_sequences or (
    seqtype_idx,
    seqidx,
  ) in posetrack18_testval_sequences:
    fname += "_test"
  else:
    assert (seqtype_idx, seqidx) in posetrack17_train_sequences or (
      seqtype_idx,
      seqidx,
    ) in posetrack18_train_sequences
    fname += "_train"

  return fname, frameidx


def posetrack18_fname2id(fname, frameidx):
  """Generates image id 

  Args:
    fname: name of the PoseTrack sequence
    frameidx: index of the frame within the sequence
  """
  tok = os.path.basename(fname).split("_")
  seqidx = int(tok[0])
  seqtype_idx = seqtype2idx(tok[1])

  assert frameidx >= 0 and frameidx < 1e4
  image_id = seqtype_idx * 10000000000 + seqidx * 10000 + frameidx
  return image_id


# def posetrack18_id2fname(image_id):
#   """Generates filename given image id

#   Args:
#     id: integer in the format TSSSSSSFFFF,
#       T encodes the sequence source (1: 'mpii', 2: 'bonn', 3: 'mpiinew')
#       SSSSSS is 6-digit index of the sequence
#       FFFF is 4-digit index of the image frame

#   Returns:
#     name of the video sequence
#   """
#   seqtype_idx = image_id // 100000000000
#   seqidx = (image_id % 100000000000) // 100000
#   frameidx = image_id % 100000

#   fname = "{:06}_{}".format(seqidx, idx2seqtype(seqtype_idx))

#   if (seqtype_idx, seqidx) in posetrack17_testval_sequences or (
#     seqtype_idx,
#     seqidx,
#   ) in posetrack18_testval_sequences:
#     fname += "_test"
#   else:
#     assert (seqtype_idx, seqidx) in posetrack17_train_sequences or (
#       seqtype_idx,
#       seqidx,
#     ) in posetrack18_train_sequences
#     fname += "_train"

#   return fname, frameidx


# def posetrack18_fname2id(fname, frameidx):
#   """Generates image id

#   Args:
#     fname: name of the PoseTrack sequence
#     frameidx: index of the frame within the sequence
#   """
#   tok = os.path.basename(fname).split("_")
#   seqidx = int(tok[0])
#   seqtype_idx = seqtype2idx("_".join(tok[1:]))

#   assert frameidx >= 0 and frameidx < 1e5
#   image_id = seqtype_idx * 100000000000 + seqidx * 100000 + frameidx
#   return image_id
