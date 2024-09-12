import tldextract
from urllib.parse import urlparse

def get_2ld(fqdn, part = "apex"):
    try:
        ext = tldextract.extract(fqdn)
        if len(ext.domain) == 0 or len(ext.suffix) == 0:
            return None
        else:
            if part == "tld":
                return ext.suffix
            elif part == "domain":
                return ext.domain
            else: 
                return ext.registered_domain
    except Exception as e:
        #print("ERROR - something wrong with the FQDN: {}".format(fqdn))
        #print(e)
        pass
    return None