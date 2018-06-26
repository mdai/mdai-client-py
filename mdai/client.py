class Client:

    def __init__(self, org=None, access_token=None):
        """ Web API client 
        """
        self.org = org
        self.access_token = access_token
