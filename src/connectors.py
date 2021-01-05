
"""
"""

import numpy as np
import pandas as pd
import csv

from melitk.analytics.connectors.core.authentication import Authentication
from melitk.analytics.connectors.teradata import ConnTeradata


class TeraDataConnection ():
    USER_INFO = pd.read_csv('../secrets/lara_user.TXT ', sep = "=", index_row=0, header=None)

    def fetch_current_coverage():
        tera = ConnTeradata(teradata_user, teradata_pass, auth_method=Authentication.LDAP)
        with open('./SQL/select_coverage.sql', 'r') as query_file:
             sql = query_file.read()
        res = tera.execute_response(sql)
        tera.conn.close()
        return pd.DataFrame(res)
    