import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router);

export const routes = [
    {
        path: '/',
        component: () => import("../pages/index")
    }
];

export default new Router({
    scrollerBehavior: ()=> ({ y: 0}),
    routes: routes
})
