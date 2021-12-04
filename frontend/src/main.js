import Vue from 'vue'
import App from './App.vue'
import router from './router'
import axios from 'axios'
import VueAxios from 'vue-axios'
import VueSocketIO from 'vue-socket.io'
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';

Vue.use(VueAxios, axios);
Vue.use(ElementUI);
Vue.use(new VueSocketIO({
    debug: true,
    connection: 'http://localhost:5000/',
}))

Vue.config.productionTip = false

axios.defaults.baseURL = "http://localhost:5000/";

new Vue({
  render: h => h(App),
  router
}).$mount('#app')
