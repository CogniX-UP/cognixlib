from cognixcore import Node, PortConfig 

class UtilNode(Node):
    
    def have_gui(self):
        return hasattr(self, 'gui')

class GetVarNode(UtilNode):
    """Gets the value of a script variable"""

    version = 'v0.2'

    title = 'get var'
    init_inputs = [
        PortConfig(),
    ]
    init_outputs = [PortConfig(label='val')]

    def __init__(self, params):
        super().__init__(params)

        self.var_name = ''
        self.temp_var_val = None

    def place_event(self):
        self.update()
        if self.input(0) is not None:
            self.var_name = self.input(0)

    def start(self):
        if not self.input_connected(0):
            self.update_event(0)
    
    def update_event(self, input_called=-1):
        if self.input(0) != self.var_name:
            if self.vars_addon.var_exists(self.flow, self.var_name):
                self.vars_addon.unsubscribe(self, self.var_name, self.var_val_changed)

            self.var_name = self.input(0)

            # create new var update connection
            if self.var_name and self.vars_addon.var_exists(self.flow, self.var_name):
                self.vars_addon.subscribe(self, self.var_name, self.var_val_changed)

        self.set_output(0, self.var_val_get(self.var_name))

    def var_val_changed(self, _):
        self.set_output(0, self.var_val(self.var_name))

class ResultNode(UtilNode):
    """Simply shows a value converted to str"""

    version = 'v0.2'

    title = 'result'
    init_inputs = [
        PortConfig(type_='data'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.val = None

    def rebuilt(self):
        self.update()

    def update_event(self, input_called=-1):
        self.val = self.input(0)
        self.updated.emit(0)


class ValNode(UtilNode):
    """Evaluates a string from the input field"""

    version = 'v0.2'

    title = 'val'
    init_inputs = [
        # NodeInputType(default=Data()),
    ]
    init_outputs = [
        PortConfig(),
    ]

    def __init__(self, params):
        super().__init__(params)

        self.display_title = ''
        self.val = None

    def init(self):
        self.update_event(0)
        
    def place_event(self):
        self.update()
    
    def start(self):
        self.update_event(0)
        
    def update_event(self, input_called=-1):
        self.set_output(0, self.val)

    def get_current_var_name(self):
        return self.input(0) if self.input(0) is not None else None

    def get_state(self):
        return {'val': self.val}  # self.main_widget().get_val()

    def set_state(self, data, version):
        self.val = data['val']


class SetVarNode(UtilNode):
    """Sets the value of a script variable"""

    version = 'v0.1'

    title = 'set var'
    init_inputs = [
        PortConfig(type_='exec'),
        PortConfig(label='var'),
        PortConfig(label='val'),
    ]
    init_outputs = [
        PortConfig(type_='exec'),
        PortConfig(type_='data', label='val'),
    ]

    def __init__(self, params):
        super().__init__(params)

        self.active = True

        self.var_name = ''
        self.num_vars = 1

    def place_event(self):
        if self.have_gui():
            self.gui.actions['make passive'] = {'method': self.action_make_passive}

    def update_event(self, input_called=-1):
        if self.active and input_called == 0:
            if self.set_var_val(self.input(1), self.input(2)):
                self.set_output(1, self.input(2))
            self.exec_output(0)

        elif not self.active:
            self.var_name = self.input(0)
            if self.set_var_val(self.input(0), self.input(1)):
                self.set_output(0, self.var_val(self.var_name))

    def action_make_passive(self):
        self.active = False
        self.delete_input(0)
        self.delete_output(0)
        del self.gui.actions['make passive']
        self.gui.actions['make active'] = {'method': self.action_make_active}

    def action_make_active(self):
        self.active = True
        self.create_input(type_='exec', pos=0)
        self.create_output(type_='exec', pos=0)
        del self.gui.actions['make active']
        self.gui.actions['make passive'] = {'method': self.action_make_passive}

    def get_state(self):
        return {'active': self.active}

    def set_state(self, data, version):
        self.active = data['active']

        # because otherwise he widgets won't work
        if not self.active:
            self.action_make_passive()


class SetVarsPassiveNode(UtilNode):
    """Sets the values of multiple script variables"""

    version = 'v0.1'

    title = 'set vars passive'
    init_inputs = []
    init_outputs = []

    def __init__(self, params):
        super().__init__(params)

        self.num_vars = 0

    def add_var_input(self):
        # self.create_input_dt(label='var', dtype=dtypes.String(size='l'))
        # self.create_input_dt(label='val', dtype=dtypes.Data(size='l'))
        self.create_input(label='var')
        self.create_input(label='val')

        self.num_vars += 1

        if self.have_gui():
            self.gui.rebuild_remove_actions()

    def remove_var_input(self, number):
        self.delete_input((number - 1) * 2)
        self.delete_input((number - 1) * 2)
        self.num_vars -= 1
        # self.rebuild_remove_actions()

        if self.have_gui():
            self.gui.rebuild_remove_actions()

    def update_event(self, input_called=-1):
        var_names = [self.input(i)for i in range(0, len(self._inputs), 2)]
        values = [self.input(i) for i in range(1, len(self._inputs), 2)]

        for i in range(len(var_names)):
            self.set_var_val(var_names[i], values[i])

    def get_state(self):
        return {'num vars': self.num_vars}

    def set_state(self, data, version):
        self.num_vars = data['num vars']