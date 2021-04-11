from pytorch_transformers import BertTokenizer

from emt.config import Config
from emt.data_representation import InputExample
from emt.feature_extraction import convert_examples_to_features

if __name__ == "__main__":
    sample1_a = "sources . please help improve this article by adding citations to reliable sources . unsourced material may be challenged and removed . july 2016 learn how and when to remove this template message bw group limited type private industry shipping headquarters oslo and singapore area served global key people chairman andreas sohmen pao ; ceo carsten mortensen number of employees 4 500 2016 website www.bwgroup.net the bw group previously bergesen worldwide is an international maritime group that through its subsidiaries operates within the tanker gas and offshore segments with a fleet of more than 100 owned part owned or controlled vessels . the group has seven business areas crude tankers product tankers chemical tankers lng liquefied natural gas carriers lpg liquefied petroleum gas carriers offshore floating production vessels fpso and offshore technology mooring systems for fpsos and offshore lng terminals . the company was created as a merger between bergesen d.y. and world wide shipping . operations are performed by subsidiaries with bw gas and bw offshore being publicly listed companies . bw gas delisted in 2009 . main offices are in oslo and singapore while the holding company is registered in bermuda . the bw group comprises bw offshore bw lpg bw pacific bw lng bw vlcc bw chemical tankers bw dry cargo and bw fleet management . with offices in eight countries bw group employs more than 4 500 people . bw group manages the world s largest independently owned and operated fleet of gas carriers lng and lpg and the second largest fleet of fpso vessels . history edit the bw group includes two long standing shipping businesses world wide shipping founded by sir yue kong pao and bergesen d.y. which was acquired by world wide in 2003 . in the late 1970s two british hongs were taken over by chinese entrepreneurs hutchison whampoa ltd by li ka shing and kowloon wharf by yue kong pao . y.k. pao was a native from ningbo zhejiang province who came to hong kong in 1949 and acquired an aging ship to establish the worldwide shipping company in 1955 . at its peak in 1979 world wide had 204 ships totalling 20 500 000 tonnes deadweight dwt . in the shipping downturn starting in 1978 world wide sold tonnage while prices were still reasonable paying off debt and building cash resources . in less than five years the fleet had halved in size allowing world wide to avoid the crises suffered by many shipping companies . in 1989 pao passed on his interest in world wide to his eldest daughter anna sohmen pao shortly before his death in 1991 . world wide was ranked as the sixth largest shipping fleet number of vessels in the world in 2004 and has been chaired by helmut sohmen pao s eldest daughter s husband since 1986 . in late 2000 following a period of stakebuilding starting in bergesen d.y. the sohmen family came to an agreement in march 2003 with morten bergesen and petter sundt the controlling family shareholders to buy their interest in bergesen and mount a full public offering for all outstanding shares . in 2004 following completion of this transaction there was a reorganisation of the sohmen family interests with the creation of a new top holding company bergesen worldwide ltd. registered in bermuda which is owned 93 % by the sohmen family and 7 % by hsbc . the holding company was subsequently renamed bw group limited . in 2005 a further re organisation took place accompanied by the re branding of the business under a single group brand bw . bw gas was listed in oslo in october 2005 as a pure gas shipping company with the bw group retaining a majority share . bw offshore followed suit by being listed in may 2006 . the other assets remain in private ownership . in 2009 bw gas was privatised . on 21 november 2013 the lpg business was listed on the oslo stock exchange as bw lpg . external links edit bw group website bw lpg website bw offshore website v t e bw group companies bw group bw offshore bw lpg bw maritime bw green marine capital bw shipping and gas solutions bw fleet management defunct companies bergesen d.y. asa world wide shipping people sigval bergesen d.y. yue kong pao helmut sohmen see also category bw group retrieved from https en.wikipedia.org w index.php title bw group & oldid 730654075 categories bw groupshipping companies of norwayshipping companies of hong kongcompanies established in 20032003 establishments in norwaygas shipping companiesfloating production storage and offloading vessel operatorshidden categories articles lacking sources from july 2016all articles lacking sources navigation menu personal tools not logged intalkcontributionscreate accountlog in namespaces article talk variants views read edit view history more search navigation main pagecontentsfeatured contentcurrent eventsrandom articledonate to wikipediawikipedia store interaction helpabout wikipediacommunity portalrecent changescontact page tools what links hererelated changesupload filespecial pagespermanent linkpage informationwikidata itemcite this page print export create a bookdownload as pdfprintable version languages fran aisnorsk bokm l edit links this page was last modified on 20 july 2016 at 12 53 . text is available under the creative commons attribution sharealike license ; additional terms may apply . by using this site you agree to the terms of use and privacy policy . wikipedia is a registered trademark of the wikimedia foundation inc. a non profit organization . privacy policy about wikipedia disclaimers contact wikipedia developers cookie statement mobile view"
    sample1_b = "pacificbw lngbw vlccbw chemical tankersbw dry cargobw fleet managementnewspress releaseslatest updatesworld horizonimage galleryinvestorsbw lpg investor relationsbw offshore investor relationsregister for accesscontact ussingaporenorwayusabw dry cargobw pacific denmarkbw pacific usa featured bw offshore bw offshore is a leading global provider of floating production services to the oil and gas industry . read more bw lpg bw lpg owns and operates the world s largest fleet of liquefied petroleum gas carriers . read more bw pacific we own and operate a fleet of well maintained long range 1 and medium range product tankers . read more bw lng comprising two business units our shipping unit manages a fleet of lng carriers to bring lng to where it is needed . our gas solutions unit develop own and operate floating gas infrastructure . read more bw vlcc our fleet of vlccs ship oil to where it is needed safely and cost effectively . read more bw chemical tankers our fleet of chemical tankers are commercially managed by womar in which bw has a 50 per cent stake . read more bw dry cargo we are a newly established department looking at opportunities available in the market with particular focus on current vessels on water in the bulk segment between 50 000 and 90 000 dwt . read more bw fleet management our fleet of over 150 vessels is managed by our own team of experienced technical and marine personnel who also oversee our new building activities . read more about us bw group is a leading global maritime group involved in shipping floating gas infrastructure and deepwater oil & gas production and has been delivering energy and other vital commodities for more than 75 years with a current fleet of 168 ships . the group was founded by sir yk pao in hong kong in 1955 as world wide shipping . in 2003 the group acquired bergesen norway s largest shipping company founded in 1935 and in 2005 the business was re branded as bw . today bw group operates the world s largest gas shipping fleet lng and lpg combined with a total of 70 large gas vessels including three fsrus floating storage and regasification units . bw offshore operates the world s second largest floating oil and gas production fleet fpsos with 15 units in us brazil mexico west africa north sea and australasia . bw s fleet also includes crude oil supertankers refined oil tankers chemical tankers and dry bulk carriers . learn more latest updates 4 october 2016 bw group appoints new cfo 3 october 2016 amended offer terms and update on the voluntary offer for aurora lpg 21 september 2016 bw lpg announces acquisition of shares & decision to launch voluntary offer 22 august 2016 bw selected as fsru supplier by pakistan gasport limited 18 august 2016 are you smarter than a seafarer 7 june 2016 bw group announces resignation of group cfo nicholas gleeson 2 june 2016 through the wardrobe and into narnia just like bankruptcy 16 march 2016 bw appoints christian bonfils to lead bw dry cargo 25 january 2016 lng fsru bw singapore in full operation world horizon world horizon is a quarterly publication by the bw group . click on the image to read more . join us our people are an important part of our success and we are always looking for talented individuals to be part of the team . click here to see a list of available vacancies . about usbw grouphistorybw group fleetprivacy statementour peopleboard of directorsmanagement committeeour valuescareerssafetysecurityemergency responsezero harm campaignsustainabilityenvironmentcsrour businessbw offshorebw lpgbw pacificbw lngbw vlccbw chemical tankersbw dry cargobw fleet managementnewspress releaseslatest updatesworld horizonimage galleryinvestorsbw lpg investor relationsbw offshore investor relationsregister for access 2016 bw group all rights reserved for best viewing experience please use ie 8 and above"

    example = InputExample("", text_a=sample1_a, text_b=sample1_b, label= 1)

    tokenizer = BertTokenizer.from_pretrained(Config().PRE_TRAINED_MODEL_BERT_BASE_UNCASED,
                                              do_lower_case=Config().DO_LOWER_CASE)

    convert_examples_to_features([example], [0, 1], 512, tokenizer)

