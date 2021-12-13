import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router);

export const routes = [
    {
        path: '/',
        redirect: '/filters'
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
                path: '/gradients',
                component: () => import('../pages/gradients'),
                meta: { title: 'Gradients' }
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
            {
                path: '/intergratedgrad',
                component: () => import('../pages/intergratedgrad'),
                meta: { title: 'IntergratedGrad' }
            },
            {
                path: '/accumulated_grad',
                component: () => import('../pages/accumulated_grad'),
                meta: { title: 'AccumulatedGrad' }
            },
            {
                path: '/relative_grad',
                component: () => import('../pages/relative_grad'),
                meta: { title: 'RelativeGrad' }
            },
            {
                path: '/fgsm',
                component: () => import('../pages/fgsm'),
                meta: { title: 'FGSM' }
            },
            {
                path: '/fgsm_grad',
                component: () => import('../pages/fgsm_grad'),
                meta: { title: 'FGSM Gradients' }
            },
            {
                path: '/fgsm_grad_diff',
                component: () => import('../pages/fgsm_grad_diff'),
                meta: { title: 'FGSM Gradients Diff' }
            },
            {
                path: '/fgsm_act',
                component: () => import('../pages/fgsm_activations'),
                meta: { title: 'FGSM Activations' }
            },
            {
                path: '/fgsm_act_diff',
                component: () => import('../pages/fgsm_activations_diff'),
                meta: { title: 'FGSM Activations Diff' }
            },
        ]
    }
];

export default new Router({
    scrollerBehavior: ()=> ({ y: 0}),
    routes: routes
})
