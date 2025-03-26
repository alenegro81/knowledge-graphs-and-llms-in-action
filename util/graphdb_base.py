from neo4j import GraphDatabase
import configparser
import os
import sys
import getopt

help_message = '-u <neo4j username> -p <password> -s <source directory> -b <bolt uri>'

neo4j_user = 'neo4j'
neo4j_password = 'password'
source_dataset_path = ''
uri = 'bolt://localhost:7687'


class GraphDBBase:
    def __init__(self, command=None, argv=None, extended_options='', extended_long_options=None):
        if extended_long_options is None:
            extended_long_options = []
        self.uri = None
        self.neo4j_user = None
        self.neo4j_password = None
        self.source_dataset_path = None
        self.database = None
        self.opts = {}
        self.args = []

        if argv:
            self.__get_main_parameters__(command=command, argv=argv, extended_options=extended_options,
                                         extended_long_options=extended_long_options)

        config = configparser.ConfigParser()
        config_file = os.path.join(os.path.dirname(__file__), '..', 'config.ini')
        config.read(config_file)
        neo4j_params = config['neo4j']

        uri = self.uri or os.getenv('NEO4J_URI') or neo4j_params.get('uri', 'bolt://localhost:7687')
        user = self.neo4j_user or os.getenv('NEO4J_USER') or neo4j_params.get('user', 'neo4j')
        password = self.neo4j_password or os.getenv('NEO4J_PASSWORD') or neo4j_params.get('password', 'password')
        self.database = self.database or os.getenv('NEO4J_DATABASE') or neo4j_params.get('database', 'neo4j')
        ignored_params = {'uri', 'user', 'password'}
        param_converters = {'encrypted': lambda x: int(x)}

        def maybe_convert(key: str, value: str):
            if key in param_converters:
                return param_converters[key](value)
            return value

        other_params = dict([(key, maybe_convert(key, value)) for key, value in neo4j_params.items()
                             if key not in ignored_params])
        # print(other_params)

        self._driver = GraphDatabase.driver(uri, auth=(user, password), **other_params)
        self._session = None

    def get_opts(self):
        return self.opts

    def get_option(self, options: list, default = None):
        for opt, arg in self.opts:
            if opt in options:
                return arg

        return default

    def close(self):
        self._driver.close()

    def get_session(self):
        return self._driver.session(database=self.database)

    def execute_without_exception(self, query: str):
        try:
            self.get_session().run(query)
        except Exception as e:
            pass

    def executeNoException(self, session, query: str):
        try:
            session.run(query)
        except Exception as e:
            pass

    def __get_main_parameters__(self, command, argv, extended_options='', extended_long_options=[]):
        try:
            self.opts, self.args = getopt.getopt(argv, 'hu:p:s:b:d:' + extended_options,
                                       ['help', 'neo4j-user=', 'neo4j-password=', 'source-path=',
                                        'bolt=', 'database='] + extended_long_options)
        except getopt.GetoptError as e:
            print(e)
            print(command, help_message)
            sys.exit(2)
        for opt, arg in self.opts:
            if opt == '-h':
                print(command, help_message)
                sys.exit()
            elif opt in ("-u", "--neo4j-user"):
                self.neo4j_user = arg
            elif opt in ("-p", "--neo4j-password"):
                self.neo4j_password = arg
            elif opt in ("-s", "--source-path"):
                self.source_dataset_path = arg
            elif opt in ("-b", "--bolt"):
                self.uri = arg
            elif opt in ("-d", "--database"):
                self.database = arg
