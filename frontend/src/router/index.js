import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router);

export const routes = [
    {
        path: '/',
        redirect: '/saliency'
    },
    {
        path: '/',
        component: () => import('../components/home'),
        children : [
            {
                path: '/activations',
                component: () => import('../pages/activations'),
                meta: { title: 'Activations' }
            },
            {
                path: '/filters',
                component: () => import('../pages/filters'),
                meta: { title: 'Filters' }
            },
            {
                path: '/deep_dream',
                component: () => import('../pages/deep_dream'),
                meta: { title: 'Deep Dream' }
            },
            {
                path: '/class_max',
                component: () => import('../pages/class_max'),
                meta: { title: 'Class maximum' }
            },
            {
                path: '/activation_max',
                component: () => import('../pages/act_max'),
                meta: { title: 'Activation Maximum' }
            },
            {
                path: '/saliency',
                component: () => import('../pages/saliency'),
                meta: { title: 'Saliency Map' }
            },
            {
                path: '/smooth_grad',
                component: () => import('../pages/smooth_grad'),
                meta: { title: 'SmoothGrad' }
            },
            {
                path: '/gradcam',
                component: () => import('../pages/grad_cam'),
                meta: { title: 'GradCAM' }
            },
        ]
    }
];

export default new Router({
    scrollerBehavior: ()=> ({ y: 0}),
    routes: routes
})
